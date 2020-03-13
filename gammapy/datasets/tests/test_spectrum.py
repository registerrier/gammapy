# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from gammapy.data import GTI
from gammapy.datasets import Datasets, SpectrumDataset, SpectrumDatasetOnOff
from gammapy.irf import EDispKernel, EffectiveAreaTable
from gammapy.maps import MapAxis, WcsGeom, RegionNDMap, RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    ConstantSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.random import get_random_state
from gammapy.utils.regions import compound_region_to_list, make_region
from gammapy.utils.testing import (
    assert_time_allclose,
    mpl_plot_check,
    requires_data,
    requires_dependency,
)
from gammapy.utils.time import time_ref_to_dict


@pytest.fixture()
def spectrum_dataset():
    energy = np.logspace(-1, 1, 31) * u.TeV
    livetime = 100 * u.s

    pwl = PowerLawSpectralModel(
            index=2.1,
            amplitude="1e5 cm-2 s-1 TeV-1",
            reference="0.1 TeV",
        )

    model = SkyModel(spectral_model=pwl, name="test-source")
    aeff = EffectiveAreaTable.from_constant(energy, "1 cm2")

    axis = MapAxis.from_edges(energy, interp="log", name="energy")
    background = RegionNDMap.create(region="icrs;circle(0, 0, 0.1)", axes=[axis])

    bkg_rate = np.ones(30) / u.s
    background.quantity = bkg_rate * livetime

    dataset = SpectrumDataset(
        models=model,
        aeff=aeff,
        livetime=livetime,
        background=background,
        name="test",
    )
    dataset.fake(random_state=23)
    return dataset


def test_data_shape(spectrum_dataset):
    assert spectrum_dataset.data_shape[0] == 30


def test_str(spectrum_dataset):
    assert "SpectrumDataset" in str(spectrum_dataset)


def test_energy_range(spectrum_dataset):
    energy_range = spectrum_dataset.energy_range
    assert energy_range.unit == u.TeV
    assert_allclose(energy_range.to_value("TeV"), [0.1, 10.0])


def test_info_dict(spectrum_dataset):
    info_dict = spectrum_dataset.info_dict()

    assert_allclose(info_dict["n_on"], 907010)
    assert_allclose(info_dict["background"], 3000.0)

    assert_allclose(info_dict["significance"], 2924.522174)
    assert_allclose(info_dict["excess"], 904010)
    assert_allclose(info_dict["livetime"].value, 1e2)

    assert info_dict["name"] == "test"


def test_incorrect_mask(spectrum_dataset):
    mask_fit = np.ones(30, dtype=np.dtype("float"))
    with pytest.raises(ValueError):
        SpectrumDataset(
            counts=spectrum_dataset.counts.copy(),
            livetime="1h",
            mask_fit=mask_fit,
        )


def test_set_model(spectrum_dataset):
    spectrum_dataset = spectrum_dataset.copy()
    spectral_model = PowerLawSpectralModel()
    model = SkyModel(spectral_model=spectral_model, name="test")
    spectrum_dataset.models = model
    assert spectrum_dataset.models["test"] is model

    models = Models([model])
    spectrum_dataset.models = models
    assert spectrum_dataset.models["test"] is model


def test_npred_models():
    e_reco = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3).edges
    spectrum_dataset = SpectrumDataset.create(e_reco=e_reco)
    spectrum_dataset.livetime = 1 * u.h
    spectrum_dataset.aeff.data.data += 1e10 * u.Unit("cm2")

    pwl_1 = PowerLawSpectralModel(index=2)
    pwl_2 = PowerLawSpectralModel(index=2)
    model_1 = SkyModel(spectral_model=pwl_1)
    model_2 = SkyModel(spectral_model=pwl_2)

    spectrum_dataset.models = Models([model_1, model_2])

    npred = spectrum_dataset.npred()

    assert_allclose(npred.data.sum(), 64.8)


@requires_dependency("iminuit")
def test_fit(spectrum_dataset):
    """Simple CASH fit to the on vector"""
    fit = Fit([spectrum_dataset])
    result = fit.run()

    # assert result.success
    assert "minuit" in repr(result)

    npred = spectrum_dataset.npred().data.sum()
    assert_allclose(npred, 907012.186399, rtol=1e-3)
    assert_allclose(result.total_stat, -18087404.624, rtol=1e-3)

    pars = result.parameters
    assert_allclose(pars["index"].value, 2.1, rtol=1e-2)
    assert_allclose(pars.error("index"), 0.00127, rtol=1e-2)

    assert_allclose(pars["amplitude"].value, 1e5, rtol=1e-3)
    assert_allclose(pars.error("amplitude"), 153.450, rtol=1e-2)


def test_spectrum_dataset_create():
    e_reco = u.Quantity([0.1, 1, 10.0], "TeV")
    e_true = u.Quantity([0.05, 0.5, 5, 20.0], "TeV")
    empty_spectrum_dataset = SpectrumDataset.create(e_reco, e_true, name="test")

    assert empty_spectrum_dataset.name == "test"
    assert empty_spectrum_dataset.counts.data.sum() == 0
    assert empty_spectrum_dataset.data_shape[0] == 2
    assert empty_spectrum_dataset.background.data.sum() == 0
    assert empty_spectrum_dataset.background.geom.axes[0].nbin == 2
    assert empty_spectrum_dataset.aeff.data.axis("energy_true").nbin == 3
    assert empty_spectrum_dataset.edisp.data.axis("energy").nbin == 2
    assert empty_spectrum_dataset.livetime.value == 0
    assert len(empty_spectrum_dataset.gti.table) == 0
    assert empty_spectrum_dataset.energy_range[0] is None
    assert_allclose(empty_spectrum_dataset.mask_safe, 0)


def test_spectrum_dataset_stack_diagonal_safe_mask(spectrum_dataset):
    geom = spectrum_dataset.counts.geom

    energy = np.logspace(-1, 1, 31) * u.TeV
    aeff = EffectiveAreaTable.from_parametrization(energy, "HESS")
    edisp = EDispKernel.from_diagonal_response(energy, energy)
    livetime = 100 * u.s
    background = spectrum_dataset.background
    spectrum_dataset1 = SpectrumDataset(
        counts=spectrum_dataset.counts.copy(),
        livetime=livetime,
        aeff=aeff,
        edisp=edisp,
        background=background.copy()
    )

    livetime2 = 0.5 * livetime
    aeff2 = EffectiveAreaTable(
        energy[:-1], energy[1:], 2 * aeff.data.data
    )
    bkg2 = RegionNDMap.from_geom(geom=geom, data=2 * background.data)

    geom = spectrum_dataset.counts.geom
    data = np.ones(spectrum_dataset.data_shape, dtype="bool")
    data[0] = False
    safe_mask2 = RegionNDMap.from_geom(geom=geom, data=data)

    spectrum_dataset2 = SpectrumDataset(
        counts=spectrum_dataset.counts.copy(),
        livetime=livetime2,
        aeff=aeff2,
        edisp=edisp,
        background=bkg2,
        mask_safe=safe_mask2,
    )
    spectrum_dataset1.stack(spectrum_dataset2)

    reference = spectrum_dataset.counts.data
    assert_allclose(spectrum_dataset1.counts.data[1:], reference[1:] * 2)
    assert_allclose(spectrum_dataset1.counts.data[0], 141363)
    assert spectrum_dataset1.livetime == 1.5 * livetime
    assert_allclose(spectrum_dataset1.background.data[1:], 3 * background.data[1:])
    assert_allclose(spectrum_dataset1.background.data[0], background.data[0])
    assert_allclose(
        spectrum_dataset1.aeff.data.data.to_value("m2"),
        4.0 / 3 * aeff.data.data.to_value("m2"),
    )
    assert_allclose(spectrum_dataset1.edisp.pdf_matrix[1:], edisp.pdf_matrix[1:])
    assert_allclose(spectrum_dataset1.edisp.pdf_matrix[0], 0.5 * edisp.pdf_matrix[0])


def test_spectrum_dataset_stack_nondiagonal_no_bkg(spectrum_dataset):
    energy = spectrum_dataset.counts.geom.axes[0].edges

    aeff = EffectiveAreaTable.from_parametrization(energy, "HESS")
    edisp1 = EDispKernel.from_gauss(energy, energy, 0.1, 0)
    livetime = 100 * u.s
    spectrum_dataset1 = SpectrumDataset(
        counts=spectrum_dataset.counts.copy(),
        livetime=livetime, aeff=aeff, edisp=edisp1,
    )

    livetime2 = livetime
    aeff2 = EffectiveAreaTable(
        energy[:-1], energy[1:], aeff.data.data
    )
    edisp2 = EDispKernel.from_gauss(energy, energy, 0.2, 0.0)
    spectrum_dataset2 = SpectrumDataset(
        counts=spectrum_dataset.counts.copy(),
        livetime=livetime2,
        aeff=aeff2,
        edisp=edisp2,
    )
    spectrum_dataset1.stack(spectrum_dataset2)

    assert spectrum_dataset1.background is None
    assert spectrum_dataset1.livetime == 2 * livetime
    assert_allclose(
        spectrum_dataset1.aeff.data.data.to_value("m2"), aeff.data.data.to_value("m2")
    )
    assert_allclose(spectrum_dataset1.edisp.get_bias(1 * u.TeV), 0.0, atol=1.2e-3)
    assert_allclose(spectrum_dataset1.edisp.get_resolution(1 * u.TeV), 0.1581, atol=1e-2)


@requires_dependency("matplotlib")
def test_peek(spectrum_dataset):
    with mpl_plot_check():
        spectrum_dataset.peek()

    with mpl_plot_check():
        spectrum_dataset.plot_fit()

    spectrum_dataset.edisp = None
    with mpl_plot_check():
        spectrum_dataset.peek()


class TestSpectrumOnOff:
    """ Test ON OFF SpectrumDataset"""

    def setup(self):
        etrue = np.logspace(-1, 1, 10) * u.TeV
        self.e_true = etrue
        ereco = np.logspace(-1, 1, 5) * u.TeV
        elo = ereco[:-1]
        ehi = ereco[1:]
        self.e_reco = ereco
        self.aeff = EffectiveAreaTable(etrue[:-1], etrue[1:], np.ones(9) * u.cm ** 2)
        self.edisp = EDispKernel.from_diagonal_response(etrue, ereco)

        start = u.Quantity([0], "s")
        stop = u.Quantity([1000], "s")
        time_ref = Time("2010-01-01 00:00:00.0")
        self.gti = GTI.create(start, stop, time_ref)
        self.livetime = self.gti.time_sum

        self.on_region = make_region("icrs;circle(0.,1.,0.1)")
        off_region = make_region("icrs;box(0.,1.,0.1, 0.2,30)")
        self.off_region = off_region.union(
            make_region("icrs;box(-1.,-1.,0.1, 0.2,150)")
        )
        self.wcs = WcsGeom.create(npix=300, binsz=0.01, frame="icrs").wcs

        self.meta={"entry_1": 0., "entry_2": "str"}

        data = np.ones(elo.shape)
        data[-1] = 0  # to test stats calculation with empty bins

        axis = MapAxis.from_edges(ereco, name="energy", interp="log")
        self.on_counts = RegionNDMap.create(
            region=self.on_region, wcs=self.wcs, axes=[axis]
        )
        self.on_counts.data += 1
        self.on_counts.data[-1] = 0

        self.off_counts = RegionNDMap.create(
            region=self.off_region, wcs=self.wcs, axes=[axis]
        )
        self.off_counts.data += 10

        acceptance = RegionNDMap.from_geom(self.on_counts.geom)
        acceptance.data += 1

        data = np.ones(elo.shape)
        data[-1] = 0

        acceptance_off = RegionNDMap.from_geom(self.off_counts.geom)
        acceptance_off.data += 10

        self.dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            aeff=self.aeff,
            edisp=self.edisp,
            livetime=self.livetime,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            name="test",
            gti=self.gti,
            meta=self.meta,
        )

    def test_spectrumdatasetonoff_create(self):
        e_reco = u.Quantity([0.1, 1, 10.0], "TeV")
        e_true = u.Quantity([0.05, 0.5, 5, 20.0], "TeV")
        empty_dataset = SpectrumDatasetOnOff.create(e_reco, e_true)

        assert empty_dataset.counts.data.sum() == 0
        assert empty_dataset.data_shape[0] == 2
        assert empty_dataset.counts_off.data.sum() == 0
        assert empty_dataset.counts_off.geom.axes[0].nbin == 2
        assert_allclose(empty_dataset.acceptance_off, 1)
        assert_allclose(empty_dataset.acceptance, 1)
        assert empty_dataset.acceptance.data.shape[0] == 2
        assert empty_dataset.acceptance_off.data.shape[0] == 2
        assert empty_dataset.livetime.value == 0
        assert len(empty_dataset.gti.table) == 0
        assert empty_dataset.energy_range[0] is None

    def test_create_stack(self):
        stacked = SpectrumDatasetOnOff.create(self.e_reco, self.e_true)
        stacked.stack(self.dataset)
        assert_allclose(stacked.energy_range.value, self.dataset.energy_range.value)

    def test_alpha(self):
        assert self.dataset.alpha.data.shape == (4, 1, 1)
        assert_allclose(self.dataset.alpha.data, 0.1)

    def test_npred_no_edisp(self):
        const = 1 * u.Unit("cm-2 s-1 TeV-1")
        model = SkyModel(spectral_model=ConstantSpectralModel(const=const))
        livetime = 1 * u.s

        e_reco = self.on_counts.geom.axes[0].edges
        aeff = EffectiveAreaTable(e_reco[:-1], e_reco[1:], np.ones(4) * u.cm ** 2)
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            aeff=aeff,
            models=model,
            livetime=livetime,
        )

        energy = aeff.energy.edges
        expected = aeff.data.data[0] * (energy[-1] - energy[0]) * const * livetime

        assert_allclose(dataset.npred_sig().data.sum(), expected.value)

    @requires_dependency("matplotlib")
    def test_peek(self):
        dataset = self.dataset.copy()
        dataset.models = SkyModel(spectral_model=PowerLawSpectralModel())

        with mpl_plot_check():
            dataset.peek()

    @requires_dependency("matplotlib")
    def test_plot_fit(self):
        dataset = self.dataset.copy()
        dataset.models = SkyModel(spectral_model=PowerLawSpectralModel())

        with mpl_plot_check():
            dataset.plot_fit()

    def test_to_from_ogip_files(self, tmp_path):
        dataset = self.dataset.copy(name="test")
        dataset.to_ogip_files(outdir=tmp_path)
        newdataset = SpectrumDatasetOnOff.from_ogip_files(tmp_path / "pha_obstest.fits")

        expected_regions = compound_region_to_list(self.off_counts.geom.region)
        regions = compound_region_to_list(newdataset.counts_off.geom.region)

        assert_allclose(self.on_counts.data, newdataset.counts.data)
        assert_allclose(self.off_counts.data, newdataset.counts_off.data)
        assert_allclose(self.edisp.pdf_matrix, newdataset.edisp.pdf_matrix)
        assert_time_allclose(newdataset.gti.time_start, dataset.gti.time_start)

        assert len(regions) == len(expected_regions)
        assert regions[0].center.is_equivalent_frame(expected_regions[0].center)
        assert_allclose(regions[1].angle, expected_regions[1].angle)

        assert newdataset.meta["entry_2"] == "str"
        assert newdataset.meta["entry_1"] == 0.

    def test_to_from_ogip_files_no_edisp(self, tmp_path):

        mask_safe = RegionNDMap.from_geom(self.on_counts.geom, dtype=bool)
        mask_safe.data += True

        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            aeff=self.aeff,
            livetime=self.livetime,
            mask_safe=mask_safe,
            acceptance=1,
            name="test",
        )
        dataset.to_ogip_files(outdir=tmp_path)
        newdataset = SpectrumDatasetOnOff.from_ogip_files(tmp_path / "pha_obstest.fits")

        assert_allclose(self.on_counts.data, newdataset.counts.data)
        assert newdataset.counts_off is None
        assert newdataset.edisp is None
        assert newdataset.gti is None

    def test_energy_mask(self):
        mask = self.dataset.counts.geom.energy_mask(emin=0.3 * u.TeV, emax=6 * u.TeV)
        desired = [False, True, True, False]
        assert_allclose(mask[:, 0, 0], desired)

        mask = self.dataset.counts.geom.energy_mask(emax=6 * u.TeV)
        desired = [True, True, True, False]
        assert_allclose(mask[:, 0, 0], desired)

        mask = self.dataset.counts.geom.energy_mask(emin=1 * u.TeV)
        desired = [False, False, True, True]
        assert_allclose(mask[:, 0, 0], desired)

    def test_str(self):
        model = SkyModel(spectral_model=PowerLawSpectralModel())
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            models=model,
            aeff=self.aeff,
            livetime=self.livetime,
            edisp=self.edisp,
            acceptance=1,
            acceptance_off=10,
        )
        assert "SpectrumDatasetOnOff" in str(dataset)
        assert "wstat" in str(dataset)

    def test_fake(self):
        """Test the fake dataset"""
        source_model = SkyModel(spectral_model=PowerLawSpectralModel())
        dataset = SpectrumDatasetOnOff(
            counts=self.on_counts,
            counts_off=self.off_counts,
            models=source_model,
            aeff=self.aeff,
            livetime=self.livetime,
            edisp=self.edisp,
            acceptance=1,
            acceptance_off=10,
        )
        real_dataset = dataset.copy()

        background = RegionNDMap.from_geom(dataset.counts.geom)
        background.data += 1
        dataset.fake(background_model=background, random_state=314)

        assert real_dataset.counts.data.shape == dataset.counts.data.shape
        assert real_dataset.counts_off.data.shape == dataset.counts_off.data.shape
        assert dataset.counts_off.data.sum() == 39
        assert dataset.counts.data.sum() == 5

    def test_info_dict(self):
        info_dict = self.dataset.info_dict()

        assert_allclose(info_dict["n_on"], 3)
        assert_allclose(info_dict["n_off"], 40)
        assert_allclose(info_dict["a_on"], 1)
        assert_allclose(info_dict["a_off"], 10)

        assert_allclose(info_dict["alpha"], 0.1)
        assert_allclose(info_dict["excess"], -1)
        assert_allclose(info_dict["livetime"].value, 1e3)

        assert info_dict["name"] == "test"


@requires_data()
@requires_dependency("iminuit")
class TestSpectralFit:
    """Test fit in astrophysical scenario"""

    def setup(self):
        path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"
        self.datasets = Datasets(
            [
                SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23523.fits"),
                SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23592.fits"),
            ]
        )

        self.pwl = SkyModel(
            spectral_model=PowerLawSpectralModel(
                index=2, amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
            )
        )

        self.ecpl = SkyModel(
            spectral_model=ExpCutoffPowerLawSpectralModel(
                index=2,
                amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
                reference=1 * u.TeV,
                lambda_=0.1 / u.TeV,
            )
        )

        # Example fit for one observation
        self.datasets[0].models = self.pwl
        self.fit = Fit([self.datasets[0]])

    def set_model(self, model):
        for obs in self.datasets:
            obs.models = model

    @requires_dependency("iminuit")
    def test_basic_results(self):
        self.set_model(self.pwl)
        result = self.fit.run()
        pars = self.fit.datasets.parameters

        assert self.pwl is self.datasets[0].models[0]

        assert_allclose(result.total_stat, 38.343, rtol=1e-3)
        assert_allclose(pars["index"].value, 2.817, rtol=1e-3)
        assert pars["amplitude"].unit == "cm-2 s-1 TeV-1"
        assert_allclose(pars["amplitude"].value, 5.142e-11, rtol=1e-3)
        assert_allclose(self.datasets[0].npred().data[60], 0.6102, rtol=1e-3)
        pars.to_table()

    def test_basic_errors(self):
        self.set_model(self.pwl)
        result = self.fit.run()
        pars = result.parameters

        assert_allclose(pars.error("index"), 0.156045, rtol=1e-3)
        assert_allclose(pars.error("amplitude"), 6.473577e-12, rtol=1e-3)
        pars.to_table()

    def test_ecpl_fit(self):
        self.set_model(self.ecpl)
        fit = Fit([self.datasets[0]])
        fit.run()

        actual = fit.datasets.parameters["lambda_"].quantity
        assert actual.unit == "TeV-1"
        assert_allclose(actual.value, 0.145215, rtol=1e-2)

    def test_joint_fit(self):
        self.set_model(self.pwl)
        fit = Fit(self.datasets)
        fit.run()
        actual = fit.datasets.parameters["index"].value
        assert_allclose(actual, 2.7806, rtol=1e-3)

        actual = fit.datasets.parameters["amplitude"].quantity
        assert actual.unit == "cm-2 s-1 TeV-1"
        assert_allclose(actual.value, 5.200e-11, rtol=1e-3)

    def test_stats(self):
        dataset = self.datasets[0]
        dataset.models = self.pwl

        fit = Fit([dataset])
        result = fit.run()

        stats = dataset.stat_array()
        actual = np.sum(stats[dataset.mask_safe])

        desired = result.total_stat
        assert_allclose(actual, desired)

    def test_fit_range(self):
        # Fit range not restriced fit range should be the thresholds
        obs = self.datasets[0]
        actual = obs.energy_range[0]

        assert actual.unit == "keV"
        assert_allclose(actual.value, 8.912509e08)

    def test_no_edisp(self):
        dataset = self.datasets[0].copy()

        # Bring aeff in RECO space
        energy = dataset.counts.geom.axes[0].center
        data = dataset.aeff.data.evaluate(energy_true=energy)
        e_edges = dataset.counts.geom.axes[0].edges

        dataset.aeff = EffectiveAreaTable(
            data=data, energy_lo=e_edges[:-1], energy_hi=e_edges[1:]
        )
        dataset.edisp = None
        dataset.models = self.pwl

        fit = Fit([dataset])
        result = fit.run()
        assert_allclose(result.parameters["index"].value, 2.7961, atol=0.02)

    def test_stacked_fit(self):
        dataset = self.datasets[0].copy()
        dataset.stack(self.datasets[1])
        dataset.models = SkyModel(PowerLawSpectralModel())

        fit = Fit([dataset])
        result = fit.run()
        pars = result.parameters

        assert_allclose(pars["index"].value, 2.7767, rtol=1e-3)
        assert u.Unit(pars["amplitude"].unit) == "cm-2 s-1 TeV-1"
        assert_allclose(pars["amplitude"].value, 5.191e-11, rtol=1e-3)


def _read_hess_obs():
    path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"
    obs1 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23523.fits")
    obs2 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23592.fits")
    return [obs1, obs2]


def make_gti(times, time_ref="2010-01-01"):
    meta = time_ref_to_dict(time_ref)
    table = Table(times, meta=meta)
    return GTI(table)


@requires_data("gammapy-data")
def make_observation_list():
    """obs with dummy IRF"""
    nbin = 3
    energy = np.logspace(-1, 1, nbin + 1) * u.TeV
    livetime = 2 * u.h
    data_on = np.arange(nbin)
    dataoff_1 = np.ones(3)
    dataoff_2 = np.ones(3) * 3
    dataoff_1[1] = 0
    dataoff_2[1] = 0

    axis = MapAxis.from_edges(energy, name="energy", interp="log")
    geom = RegionGeom(region=None, axes=[axis])

    on_vector = RegionNDMap.from_geom(geom=geom, data=data_on)
    off_vector1 = RegionNDMap.from_geom(geom=geom, data=dataoff_1)
    off_vector2 = RegionNDMap.from_geom(geom=geom, data=dataoff_2)
    mask_safe = RegionNDMap.from_geom(geom, dtype=bool)
    mask_safe.data += True

    aeff = EffectiveAreaTable.from_constant(energy, "1 cm2")
    edisp = EDispKernel.from_gauss(e_true=energy, e_reco=energy, sigma=0.2, bias=0)

    time_ref = Time("2010-01-01")
    gti1 = make_gti({"START": [5, 6, 1, 2], "STOP": [8, 7, 3, 4]}, time_ref=time_ref)
    gti2 = make_gti({"START": [14], "STOP": [15]}, time_ref=time_ref)

    obs1 = SpectrumDatasetOnOff(
        counts=on_vector,
        counts_off=off_vector1,
        aeff=aeff,
        edisp=edisp,
        livetime=livetime,
        mask_safe=mask_safe,
        acceptance=1,
        acceptance_off=2,
        name="1",
        gti=gti1,
    )
    obs2 = SpectrumDatasetOnOff(
        counts=on_vector,
        counts_off=off_vector2,
        aeff=aeff,
        edisp=edisp,
        livetime=livetime,
        mask_safe=mask_safe,
        acceptance=1,
        acceptance_off=4,
        name="2",
        gti=gti2,
    )

    obs_list = [obs1, obs2]
    return obs_list


@requires_data("gammapy-data")
class TestSpectrumDatasetOnOffStack:
    def setup(self):
        self.datasets = _read_hess_obs()
        # Change threshold to make stuff more interesting

        geom = self.datasets[0]._geom
        data = geom.energy_mask(emin=1.2 * u.TeV, emax=50 * u.TeV)
        self.datasets[0].mask_safe = RegionNDMap.from_geom(geom=geom, data=data)

        data = geom.energy_mask(emax=20 * u.TeV)
        self.datasets[1].mask_safe.data &= data

        self.stacked_dataset = self.datasets[0].copy()
        self.stacked_dataset.stack(self.datasets[1])

    def test_basic(self):
        obs_1, obs_2 = self.datasets

        counts1 = obs_1.counts.data[obs_1.mask_safe].sum()
        counts2 = obs_2.counts.data[obs_2.mask_safe].sum()
        summed_counts = counts1 + counts2

        stacked_counts = self.stacked_dataset.counts.data.sum()

        off1 = obs_1.counts_off.data[obs_1.mask_safe].sum()
        off2 = obs_2.counts_off.data[obs_2.mask_safe].sum()
        summed_off = off1 + off2
        stacked_off = self.stacked_dataset.counts_off.data.sum()

        assert summed_counts == stacked_counts
        assert summed_off == stacked_off

    def test_thresholds(self):
        e_min, e_max = self.stacked_dataset.energy_range

        assert e_min.unit == "keV"
        assert_allclose(e_min.value, 8.912509e08, rtol=1e-3)

        assert e_max.unit == "keV"
        assert_allclose(e_max.value, 4.466836e10, rtol=1e-3)

    def test_verify_npred(self):
        """Veryfing npred is preserved during the stacking"""
        pwl = SkyModel(
            spectral_model=PowerLawSpectralModel(
                index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
            )
        )
        self.stacked_dataset.models = pwl

        npred_stacked = self.stacked_dataset.npred().data
        npred_stacked[~self.stacked_dataset.mask_safe.data] = 0
        npred_summed = np.zeros_like(npred_stacked)

        for obs in self.datasets:
            obs.models = pwl
            npred_summed[obs.mask_safe] += obs.npred().data[obs.mask_safe]

        assert_allclose(npred_stacked, npred_summed)

    def test_stack_backscal(self):
        """Verify backscal stacking """
        obs1, obs2 = make_observation_list()
        obs1.stack(obs2)
        assert_allclose(obs1.alpha.data[0], 1.25 / 4.0)
        # When the OFF stack observation counts=0, the alpha is averaged on the total OFF counts for each run.
        assert_allclose(obs1.alpha.data[1], 2.5 / 8.0)

    def test_stack_gti(self):
        obs1, obs2 = make_observation_list()
        obs1.stack(obs2)
        table_gti = Table({"START": [1.0, 5.0, 14.0], "STOP": [4.0, 8.0, 15.0]})
        table_gti_stacked_obs = obs1.gti.table
        assert_allclose(table_gti_stacked_obs["START"], table_gti["START"])
        assert_allclose(table_gti_stacked_obs["STOP"], table_gti["STOP"])


@requires_data("gammapy-data")
def test_datasets_stack_reduce():
    obs_ids = [23523, 23526, 23559, 23592]
    dataset_list = []
    for obs in obs_ids:
        filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{}.fits"
        ds = SpectrumDatasetOnOff.from_ogip_files(filename.format(obs))
        dataset_list.append(ds)
    datasets = Datasets(dataset_list)
    stacked = datasets.stack_reduce(name="stacked")
    assert_allclose(stacked.livetime.to_value("s"), 6313.8116406202325)

    info_table = datasets.info_table()
    assert_allclose(info_table["n_on"], [124, 126, 119, 90])

    info_table_cum = datasets.info_table(cumulative=True)
    assert_allclose(info_table_cum["n_on"], [124, 250, 369, 459])
    assert stacked.name == "stacked"


def test_spectrum_dataset_on_off_to_yaml(tmpdir):
    spectrum_datasets_on_off = make_observation_list()
    datasets = Datasets(spectrum_datasets_on_off)
    datasets.write(path=tmpdir)
    datasets_read = Datasets.read(tmpdir / "_datasets.yaml", tmpdir / "_models.yaml")
    assert len(datasets_read) == len(datasets)
    assert datasets_read[0].name == datasets[0].name
    assert datasets_read[1].name == datasets[1].name
    assert datasets_read[1].counts.data.sum() == datasets[1].counts.data.sum()


def test_stack_no_livetime():
    e_reco = np.logspace(0, 1, 3) * u.TeV
    dataset_1 = SpectrumDataset.create(e_reco=e_reco)
    dataset_1.livetime = None
    dataset_2 = dataset_1.copy()

    with pytest.raises(ValueError):
        dataset_1.stack(dataset_2)


@requires_dependency("iminuit")
class TestFit:
    """Test fit on counts spectra without any IRFs"""

    def setup(self):
        self.nbins = 30
        energy = np.logspace(-1, 1, self.nbins + 1) * u.TeV
        self.source_model = SkyModel(
            spectral_model=PowerLawSpectralModel(
                index=2, amplitude=1e5 * u.Unit("cm-2 s-1 TeV-1"), reference=0.1 * u.TeV
            )
        )
        bkg_model = PowerLawSpectralModel(
            index=3, amplitude=1e4 * u.Unit("cm-2 s-1 TeV-1"), reference=0.1 * u.TeV
        )

        self.alpha = 0.1
        random_state = get_random_state(23)
        npred = self.source_model.spectral_model.integral(
            energy[:-1], energy[1:]
        ).value
        source_counts = random_state.poisson(npred)

        axis = MapAxis.from_edges(energy, name="energy", interp="log")
        geom = RegionGeom(region=None, axes=[axis])

        self.src = RegionNDMap.from_geom(geom=geom, data=source_counts)

        self.src.livetime = 1 * u.s
        self.aeff = EffectiveAreaTable.from_constant(energy, "1 cm2")

        npred_bkg = bkg_model.integral(energy[:-1], energy[1:]).value

        bkg_counts = random_state.poisson(npred_bkg)
        off_counts = random_state.poisson(npred_bkg * 1.0 / self.alpha)
        self.bkg = RegionNDMap.from_geom(geom=geom, data=bkg_counts)
        self.off = RegionNDMap.from_geom(geom=geom, data=off_counts)

    def test_cash(self):
        """Simple CASH fit to the on vector"""
        dataset = SpectrumDataset(
            models=self.source_model,
            counts=self.src,
            aeff=self.aeff,
            livetime=self.src.livetime,
        )

        npred = dataset.npred().data
        assert_allclose(npred[5], 660.5171, rtol=1e-5)

        stat_val = dataset.stat_sum()
        assert_allclose(stat_val, -107346.5291, rtol=1e-5)

        self.source_model.parameters["index"].value = 1.12

        fit = Fit([dataset])
        result = fit.run()

        # These values are check with sherpa fits, do not change
        pars = result.parameters
        assert_allclose(pars["index"].value, 1.995525, rtol=1e-3)
        assert_allclose(pars["amplitude"].value, 100245.9, rtol=1e-3)

    def test_wstat(self):
        """WStat with on source and background spectrum"""
        on_vector = self.src.copy()
        on_vector.data += self.bkg.data
        dataset = SpectrumDatasetOnOff(
            counts=on_vector,
            counts_off=self.off,
            aeff=self.aeff,
            livetime=self.src.livetime,
            acceptance=1,
            acceptance_off=1 / self.alpha,
        )
        dataset.models = self.source_model

        self.source_model.parameters.index = 1.12

        fit = Fit([dataset])
        result = fit.run()
        pars = self.source_model.parameters

        assert_allclose(pars["index"].value, 1.997342, rtol=1e-3)
        assert_allclose(pars["amplitude"].value, 100245.187067, rtol=1e-3)
        assert_allclose(result.total_stat, 30.022316, rtol=1e-3)

    def test_fit_range(self):
        """Test fit range without complication of thresholds"""
        geom = self.src.geom
        mask_safe = RegionNDMap.from_geom(geom, dtype=bool)
        mask_safe.data += True

        dataset = SpectrumDatasetOnOff(
            counts=self.src, mask_safe=mask_safe
        )

        assert np.sum(dataset.mask_safe) == self.nbins
        e_min, e_max = dataset.energy_range

        assert_allclose(e_max.value, 10)
        assert_allclose(e_min.value, 0.1)

    def test_stat_profile(self):
        geom = self.src.geom
        mask_safe = RegionNDMap.from_geom(geom, dtype=bool)
        mask_safe.data += True

        dataset = SpectrumDataset(
            models=self.source_model,
            aeff=self.aeff,
            livetime=self.src.livetime,
            counts=self.src,
            mask_safe=mask_safe,
        )
        fit = Fit([dataset])
        result = fit.run()
        true_idx = result.parameters["index"].value
        values = np.linspace(0.95 * true_idx, 1.05 * true_idx, 100)
        profile = fit.stat_profile("index", values=values)
        actual = values[np.argmin(profile["stat"])]
        assert_allclose(actual, true_idx, rtol=0.01)
