# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from astropy.table import Table

from gammapy.datasets import Dataset
from gammapy.modeling import Fit, Parameter
from gammapy.modeling.fit import FitResult, OptimizeResult
from gammapy.modeling.models import ModelBase, Models
from gammapy.visualization.display import FitResultDisplay


class SimpleModel(ModelBase):
    x = Parameter("x", 2)
    y = Parameter("y", 3e2)
    z = Parameter("z", 4e-2)
    name = "simple"
    datasets_names = ["test"]
    type = "model"


class SimpleDataset(Dataset):
    tag = "SimpleDataset"

    def __init__(self):
        self._name = "test"
        self._models = Models([SimpleModel()])
        self.data_shape = (1,)
        self.meta_table = Table()

    @property
    def models(self):
        return self._models

    def stat_sum(self):
        x, y, z = [p.value for p in self.models.parameters.unique_parameters]
        return (x - 2) ** 2 + (y - 3e2) ** 2 + (z - 4e-2) ** 2

    def stat_array(self):
        pass


@pytest.fixture
def result_no_covariance():
    models = SimpleDataset().models
    opt = OptimizeResult(
        models=models,
        nfev=10,
        total_stat=0.0,
        trace=Table(),
        backend="minuit",
        method="migrad",
        success=True,
        message="Optimization terminated successfully.",
    )
    return FitResult(optimize_result=opt)


@pytest.fixture
def result_with_covariance():
    return Fit().run([SimpleDataset()])


@pytest.fixture
def result_with_frozen():
    dataset = SimpleDataset()
    dataset.models["simple"].z.frozen = True
    return Fit().run([dataset])


def test_repr_html_returns_string(result_no_covariance):
    html = FitResultDisplay(result_no_covariance)._repr_html_()
    assert isinstance(html, str)
    assert len(html) > 0


def test_repr_html_summary_fields(result_no_covariance):
    html = FitResultDisplay(result_no_covariance)._repr_html_()
    assert "True" in html
    assert "0.000" in html
    assert "Optimization terminated successfully." in html
    assert "10" in html
    assert "minuit" in html
    assert "migrad" in html


def test_no_covariance_message(result_no_covariance):
    html = FitResultDisplay(result_no_covariance)._repr_html_()
    assert "No covariance available" in html


def test_covariance_produces_image(result_with_covariance):
    html = FitResultDisplay(result_with_covariance)._repr_html_()
    assert '<img src="data:image/png;base64,' in html


def test_parameters_section_present(result_no_covariance):
    html = FitResultDisplay(result_no_covariance)._repr_html_()
    assert "Parameters" in html
    assert "x" in html


def test_frozen_parameter_styled_grey(result_with_frozen):
    display = FitResultDisplay(result_with_frozen)
    df = display._make_parameter_table()
    table_html = display._render_parameter_table(df)
    assert "color: grey" in table_html


def test_free_parameters_have_no_grey_styling(result_with_covariance):
    display = FitResultDisplay(result_with_covariance)
    df = display._make_parameter_table()
    if "frozen" in df.columns:
        df = df.copy()
        df["frozen"] = False
    table_html = display._render_parameter_table(df)
    assert "color: grey" not in table_html


def test_repr_delegates_to_str(result_no_covariance):
    display = FitResultDisplay(result_no_covariance)
    assert repr(display) == str(result_no_covariance)
