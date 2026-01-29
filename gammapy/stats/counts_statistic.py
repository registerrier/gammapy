# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import html
import numpy as np
from scipy.special import lambertw
from scipy.stats import chi2
from scipy.special import gammaln, logsumexp
from gammapy.utils.roots import find_roots
from .fit_statistics import cash, wstat

__all__ = ["WStatCountsStatistic", "CashCountsStatistic", "LStatCountsStatistic"]


class CountsStatistic(abc.ABC):
    """Counts statistics base class."""

    @property
    @abc.abstractmethod
    def stat_null(self):
        pass

    @property
    @abc.abstractmethod
    def stat_max(self):
        pass

    @property
    @abc.abstractmethod
    def n_sig(self):
        pass

    @property
    @abc.abstractmethod
    def n_bkg(self):
        pass

    @property
    @abc.abstractmethod
    def error(self):
        pass

    @abc.abstractmethod
    def _stat_fcn(self):
        pass

    @property
    def ts(self):
        """Return stat difference (TS) of measured excess versus no excess."""
        # Remove (small) negative TS due to error in root finding
        ts = np.clip(self.stat_null - self.stat_max, 0, None)
        return ts

    @property
    def sqrt_ts(self):
        """Return statistical significance of measured excess.

        The sign of the excess is applied to distinguish positive and negative fluctuations.
        """
        return np.sign(self.n_sig) * np.sqrt(self.ts)

    @property
    def p_value(self):
        """Return p_value of measured excess.

        Here the value accounts only for the positive excess significance (i.e. one-sided).
        """
        return 0.5 * chi2.sf(self.ts, 1)

    def __str__(self):
        str_ = "\t{:32}: {{n_on:.2f}} \n".format("Total counts")
        str_ += "\t{:32}: {{background:.2f}}\n".format("Total background counts")
        str_ += "\t{:32}: {{excess:.2f}}\n".format("Total excess counts")
        str_ += "\t{:32}: {{significance:.2f}}\n".format("Total significance")
        str_ += "\t{:32}: {{p_value:.3f}}\n".format("p - value")
        str_ += "\t{:32}: {{n_bins:.0f}}\n".format("Total number of bins")
        info = self.info_dict()
        info["n_bins"] = np.array(self.n_on).size
        str_ = str_.format(**info)

        return str_.expandtabs(tabsize=2)

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def info_dict(self):
        """A dictionary of the relevant quantities.

        Returns
        -------
        info_dict : dict
            Dictionary with summary information.
        """
        info_dict = {}
        info_dict["n_on"] = self.n_on
        info_dict["background"] = self.n_bkg
        info_dict["excess"] = self.n_sig
        info_dict["significance"] = self.sqrt_ts
        info_dict["p_value"] = self.p_value
        return info_dict

    def compute_errn(self, n_sigma=1.0):
        """Compute downward excess uncertainties.

        Searches the signal value for which the test statistics is n_sigma**2 away from the maximum.

        Parameters
        ----------
        n_sigma : float
            Confidence level of the uncertainty expressed in number of sigma. Default is 1.
        """
        errn = np.zeros_like(self.n_sig, dtype="float")
        min_range = self.n_sig - 2 * n_sigma * (self.error + 1)

        it = np.nditer(errn, flags=["multi_index"])
        while not it.finished:
            roots, res = find_roots(
                self._stat_fcn,
                min_range[it.multi_index],
                self.n_sig[it.multi_index],
                nbin=1,
                args=(self.stat_max[it.multi_index] + n_sigma**2, it.multi_index),
            )
            if np.isnan(roots[0]):
                errn[it.multi_index] = self.n_on[it.multi_index]
            else:
                errn[it.multi_index] = self.n_sig[it.multi_index] - roots[0]
            it.iternext()

        return errn

    def compute_errp(self, n_sigma=1):
        """Compute upward excess uncertainties.

        Searches the signal value for which the test statistics is n_sigma**2 away from the maximum.

        Parameters
        ----------
        n_sigma : float
            Confidence level of the uncertainty expressed in number of sigma. Default is 1.
        """
        errp = np.zeros_like(self.n_on, dtype="float")
        max_range = self.n_sig + 2 * n_sigma * (self.error + 1)

        it = np.nditer(errp, flags=["multi_index"])
        while not it.finished:
            roots, res = find_roots(
                self._stat_fcn,
                self.n_sig[it.multi_index],
                max_range[it.multi_index],
                nbin=1,
                args=(self.stat_max[it.multi_index] + n_sigma**2, it.multi_index),
            )
            errp[it.multi_index] = roots[0]
            it.iternext()

        return errp - self.n_sig

    def compute_upper_limit(self, n_sigma=3):
        """Compute upper limit on the signal.

        Searches the signal value for which the test statistic is n_sigma**2
        away from the maximum.


        Parameters
        ----------
        n_sigma : float
            Confidence level of the upper limit expressed in number of sigma. Default is 3.
        """
        ul = np.zeros_like(self.n_on, dtype="float")

        min_range = self.n_sig
        max_range = min_range + 2 * n_sigma * (self.error + 1)
        it = np.nditer(ul, flags=["multi_index"])

        while not it.finished:
            ts_ref = self._stat_fcn(min_range[it.multi_index], 0.0, it.multi_index)

            roots, res = find_roots(
                self._stat_fcn,
                min_range[it.multi_index],
                max_range[it.multi_index],
                nbin=1,
                args=(ts_ref + n_sigma**2, it.multi_index),
            )
            ul[it.multi_index] = roots[0]
            it.iternext()
        return ul

    @abc.abstractmethod
    def _n_sig_matching_significance_fcn(self):
        pass

    def n_sig_matching_significance(self, significance):
        """Compute excess matching a given significance.

        This function is the inverse of `significance`.

        Parameters
        ----------
        significance : float
            Significance.

        Returns
        -------
        n_sig : `numpy.ndarray`
            Excess.
        """
        n_sig = np.zeros_like(self.n_bkg, dtype="float")
        it = np.nditer(n_sig, flags=["multi_index"])

        while not it.finished:
            lower_bound = np.sqrt(self.n_bkg[it.multi_index]) * significance
            # find upper bounds for secant method as in scipy
            eps = 1e-4
            upper_bound = lower_bound * (1 + eps)
            upper_bound += eps if upper_bound >= 0 else -eps
            roots, res = find_roots(
                self._n_sig_matching_significance_fcn,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                args=(significance, it.multi_index),
                nbin=1,
                method="secant",
            )
            n_sig[it.multi_index] = roots[0]  # return NaN if fail
            it.iternext()
        return n_sig

    @abc.abstractmethod
    def sum(self, axis=None):
        """Return summed CountsStatistics.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
             Axis or axes on which to perform the summation.
             Default, axis=None, will perform the sum over the whole array.

        Returns
        -------
        stat : `~gammapy.stats.CountsStatistics`
             The summed stat object.
        """
        pass


class CashCountsStatistic(CountsStatistic):
    """Class to compute statistics for Poisson distributed variable with known background.

    Parameters
    ----------
    n_on : int
        Measured counts.
    mu_bkg : float
        Known level of background.
    """

    def __init__(self, n_on, mu_bkg):
        self.n_on = np.asanyarray(n_on)
        self.mu_bkg = np.asanyarray(mu_bkg)

    @property
    def n_bkg(self):
        """Expected background counts."""
        return self.mu_bkg

    @property
    def n_sig(self):
        """Excess."""
        return self.n_on - self.n_bkg

    @property
    def error(self):
        """Approximate error from the covariance matrix."""
        return np.sqrt(self.n_on)

    @property
    def stat_null(self):
        """Stat value for null hypothesis, i.e. 0 expected signal counts."""
        return cash(self.n_on, self.mu_bkg + 0)

    @property
    def stat_max(self):
        """Stat value for best fit hypothesis, i.e. expected signal mu = n_on - mu_bkg."""
        return cash(self.n_on, self.n_on)

    def info_dict(self):
        """A dictionary of the relevant quantities.

        Returns
        -------
        info_dict : dict
            Dictionary with summary info.
        """
        info_dict = super().info_dict()
        info_dict["mu_bkg"] = self.mu_bkg
        return info_dict

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += super().__str__()
        str_ += "\t{:32}: {:.2f} \n".format(
            "Predicted background counts", self.info_dict()["mu_bkg"]
        )
        return str_.expandtabs(tabsize=2)

    def _stat_fcn(self, mu, delta=0, index=None):
        return cash(self.n_on[index], self.mu_bkg[index] + mu) - delta

    def _n_sig_matching_significance_fcn(self, n_sig, significance, index):
        TS0 = cash(n_sig + self.mu_bkg[index], self.mu_bkg[index])
        TS1 = cash(n_sig + self.mu_bkg[index], self.mu_bkg[index] + n_sig)
        return np.sign(n_sig) * np.sqrt(np.clip(TS0 - TS1, 0, None)) - significance

    def sum(self, axis=None):
        n_on = self.n_on.sum(axis=axis)
        bkg = self.n_bkg.sum(axis=axis)
        return CashCountsStatistic(n_on=n_on, mu_bkg=bkg)

    def __getitem__(self, key):
        return CashCountsStatistic(n_on=self.n_on[key], mu_bkg=self.n_bkg[key])

    def compute_errn(self, n_sigma=1.0):
        result = np.zeros_like(self.n_on, dtype="float")
        c = n_sigma**2 / 2
        mask = self.n_on > 0
        on = self.n_on[mask]
        result[mask] = on * (lambertw(-np.exp(-c / on - 1), k=0).real + 1)
        result[~mask] = 0
        return result

    def compute_errp(self, n_sigma=1.0):
        result = np.zeros_like(self.n_on, dtype="float")
        c = n_sigma**2 / 2
        mask = self.n_on > 0

        on = self.n_on[mask]
        result[mask] = -on * (lambertw(-np.exp(-c / on - 1), k=-1).real + 1)
        result[~mask] = c
        return result

    def compute_upper_limit(self, n_sigma=3):
        result = np.zeros_like(self.n_on, dtype="float")
        c = n_sigma**2 / 2
        mask = self.n_on > 0

        on = self.n_on[mask]
        result[mask] = (
            -on * (lambertw(-np.exp(-c / on - 1), k=-1).real + 1) + self.n_sig[mask]
        )

        result[~mask] = c
        return result

    def n_sig_matching_significance(self, significance):
        result = np.zeros_like(self.mu_bkg, dtype="float")
        c = significance**2 / 2
        mask = self.mu_bkg > 0

        bkg = self.mu_bkg[mask]
        branch = 0 if significance > 0 else -1
        res = lambertw((c / bkg - 1) / np.exp(1), k=branch).real
        result[mask] = bkg * (np.exp(res + 1) - 1)
        result[~mask] = np.nan
        return result


class WStatCountsStatistic(CountsStatistic):
    """Class to compute statistics for Poisson distributed variable with unknown background.

    Parameters
    ----------
    n_on : int
        Measured counts in on region.
    n_off : int
        Measured counts in off region.
    alpha : float
        Acceptance ratio of on and off measurements.
    mu_sig : float
        Expected signal counts in on region.
    """

    def __init__(self, n_on, n_off, alpha, mu_sig=None):
        self.n_on = np.asanyarray(n_on)
        self.n_off = np.asanyarray(n_off)
        self.alpha = np.asanyarray(alpha)

        if mu_sig is None:
            self.mu_sig = np.zeros_like(self.n_on)
        else:
            self.mu_sig = np.asanyarray(mu_sig)

    @property
    def n_bkg(self):
        """Known background computed alpha * n_off."""
        return self.alpha * self.n_off

    @property
    def n_sig(self):
        """Excess."""
        return self.n_on - self.n_bkg - self.mu_sig

    @property
    def error(self):
        """Approximate error from the covariance matrix."""
        return np.sqrt(self.n_on + self.alpha**2 * self.n_off)

    @property
    def stat_null(self):
        """Stat value for null hypothesis, i.e. mu_sig expected signal counts."""
        return wstat(self.n_on, self.n_off, self.alpha, self.mu_sig)

    @property
    def stat_max(self):
        """Stat value for best fit hypothesis.

        i.e. expected signal mu = n_on - alpha * n_off - mu_sig.
        """
        return wstat(self.n_on, self.n_off, self.alpha, self.n_sig + self.mu_sig)

    def info_dict(self):
        """A dictionary of the relevant quantities.

        Returns
        -------
        info_dict : dict
            Dictionary with summary info.
        """
        info_dict = super().info_dict()
        info_dict["n_off"] = self.n_off
        info_dict["alpha"] = self.alpha
        info_dict["mu_sig"] = self.mu_sig
        return info_dict

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += super().__str__()
        info_dict = self.info_dict()
        str_ += "\t{:32}: {:.2f} \n".format("Off counts", info_dict["n_off"])
        str_ += "\t{:32}: {:.2f} \n".format("alpha ", info_dict["alpha"])
        str_ += "\t{:32}: {:.2f} \n".format(
            "Predicted signal counts", info_dict["mu_sig"]
        )
        return str_.expandtabs(tabsize=2)

    def _stat_fcn(self, mu, delta=0, index=None):
        return (
            wstat(
                self.n_on[index],
                self.n_off[index],
                self.alpha[index],
                (mu + self.mu_sig[index]),
            )
            - delta
        )

    def _n_sig_matching_significance_fcn(self, n_sig, significance, index):
        stat0 = wstat(
            n_sig + self.n_bkg[index], self.n_off[index], self.alpha[index], 0
        )
        stat1 = wstat(
            n_sig + self.n_bkg[index],
            self.n_off[index],
            self.alpha[index],
            n_sig,
        )
        return np.sign(n_sig) * np.sqrt(np.clip(stat0 - stat1, 0, None)) - significance

    def sum(self, axis=None):
        n_on = self.n_on.sum(axis=axis)
        n_off = self.n_off.sum(axis=axis)
        alpha = self.n_bkg.sum(axis=axis) / n_off
        return WStatCountsStatistic(n_on=n_on, n_off=n_off, alpha=alpha)

    def __getitem__(self, key):
        return WStatCountsStatistic(
            n_on=self.n_on[key], n_off=self.n_off[key], alpha=self.alpha[key]
        )


class LStatCountsStatistic(CountsStatistic):
    """Class to compute statistics using Loredo's Bayesian L-statistic.

    This implements the Bayesian ON-OFF Poisson statistic from Loredo (1992)
    for Poisson distributed variable with unknown background, where the
    background is marginalized over assuming uniform background distribution.

    Parameters
    ----------
    n_on : array-like
        Measured counts in on region
    n_off : array-like
        Measured counts in off region
    alpha : array-like
        Acceptance ratio of on and off measurements.
    mu_sig : array-like, optional
        Expected signal counts in on region. Default is None (equivalent to 0).
    tolerance : float, optional
        Relative tolerance for truncating sum in L-stat computation.
        Default is 1e-15.

    Notes
    -----
    The L-statistic marginalizes over the unknown background counts, providing
    a fully Bayesian treatment. This differs from the W-statistic which uses
    a profile likelihood approach (maximization over background).

    References
    ----------
    .. [1] Loredo, T. J. (1992). "The Promise of Bayesian Inference for
           Astrophysics." In Statistical Challenges in Modern Astronomy,
           Springer-Verlag, pp. 275-297.
    .. [2] XSpec documentation
    .. [3] D'amico, G. et al (2021)
    """

    def __init__(self, n_on, n_off, alpha, mu_sig=None, tolerance=1e-15):
        self.n_on = np.asanyarray(n_on, dtype=float)
        self.n_off = np.asanyarray(n_off, dtype=int)
        self.alpha = np.asanyarray(alpha, dtype=float)
        self.tolerance = tolerance

        # Beta parameter: β = 1 + 1/α
        self.beta = 1.0 + 1.0 / self.alpha

        if mu_sig is None:
            self.mu_sig = np.zeros_like(self.n_on)
        else:
            self.mu_sig = np.asanyarray(mu_sig, dtype=float)

        # Cache for log-factorials
        self._log_factorial_cache = {}

    def _log_factorial(self, n):
        """Compute log(n!) with caching and Stirling's approximation.

        Parameters
        ----------
        n : int
            Input value

        Returns
        -------
        result : float
            log of factorial n.
        """
        if isinstance(n, np.ndarray):
            return np.vectorize(self._log_factorial)(n)

        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")

        if n == 0 or n == 1:
            return 0.0

        # Use cache for small values
        if n <= 100:
            if n not in self._log_factorial_cache:
                self._log_factorial_cache[n] = gammaln(n + 1)
            return self._log_factorial_cache[n]

        # Use Stirling's approximation for large values
        return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n)

    def _lstat_term(self, mu_sig, n_on, n_off, beta):
        """Compute L-statistic contribution for a single bin.

        Parameters
        ----------
        mu_sig : float
            Expected signal counts
        n_on : int
            Observed ON counts
        n_off : int
            Observed OFF counts
        beta : float
            Parameter β = 1 + 1/α

        Returns
        -------
        float
            L-statistic contribution (2 * ℓ)
        """
        n_on = int(n_on)
        n_off = int(n_off)

        # Handle special case: no predicted signal
        if mu_sig < 1e-10:
            log_sum = self._log_factorial(n_on + n_off) - self._log_factorial(n_on)
            return 2.0 * (0.0 - log_sum)

        # Compute sum over j using incremental method
        log_terms = []

        # j=0 term
        log_term_0 = self._log_factorial(n_on + n_off) - self._log_factorial(n_on)
        log_terms.append(log_term_0)

        # Initialize for incremental computation
        current_log_term = log_term_0

        # Compute j=1 to n_on terms incrementally
        for j in range(1, n_on + 1):
            numerator = n_on - j + 1
            denominator = n_on + n_off - j + 1

            if denominator == 0:
                break

            log_ratio = (
                np.log(mu_sig)
                + np.log(beta)
                - np.log(j)
                + np.log(numerator)
                - np.log(denominator)
            )

            current_log_term += log_ratio

            # Check if term is significant
            max_log_term = max(log_terms)
            if current_log_term > max_log_term - np.log(1.0 / self.tolerance):
                log_terms.append(current_log_term)
            else:
                # Remaining terms will be even smaller
                break

        # Use log-sum-exp for numerical stability
        log_sum = logsumexp(log_terms)

        # Return 2 * [mu_sig - log(sum)]
        return 2.0 * (mu_sig - log_sum)

    def _lstat_vectorized(self, mu_sig):
        """Compute L-statistic for given signal expectation (vectorized).

        Parameters
        ----------
        mu_sig : array-like
            Expected signal counts for each bin

        Returns
        -------
        array-like
            L-statistic value for each bin
        """
        mu_sig = np.atleast_1d(mu_sig)
        result = np.zeros_like(mu_sig, dtype=float)

        # Ensure all arrays have the same shape
        n_on = np.atleast_1d(self.n_on)
        n_off = np.atleast_1d(self.n_off)
        beta = np.atleast_1d(self.beta)

        # Broadcast to common shape if needed
        if beta.shape != mu_sig.shape:
            beta = np.broadcast_to(beta, mu_sig.shape)
        if n_on.shape != mu_sig.shape:
            n_on = np.broadcast_to(n_on, mu_sig.shape)
        if n_off.shape != mu_sig.shape:
            n_off = np.broadcast_to(n_off, mu_sig.shape)

        # Flatten for iteration
        shape = mu_sig.shape
        mu_sig_flat = mu_sig.flatten()
        n_on_flat = n_on.flatten()
        n_off_flat = n_off.flatten()
        beta_flat = beta.flatten()

        for i in range(len(mu_sig_flat)):
            result.flat[i] = self._lstat_term(
                mu_sig_flat[i], n_on_flat[i], n_off_flat[i], beta_flat[i]
            )

        return result.reshape(shape)

    @property
    def n_bkg(self):
        """Known background computed as alpha * n_off"""
        return self.alpha * self.n_off

    @property
    def n_sig(self):
        """Excess: n_on - alpha * n_off - mu_sig"""
        return self.n_on - self.n_bkg - self.mu_sig

    @property
    def error(self):
        """Approximate error from the covariance matrix.

        Uses the same approximation as W-stat for consistency.
        """
        return np.sqrt(self.n_on + self.alpha**2 * self.n_off)

    @property
    def stat_null(self):
        """Stat value for null hypothesis, i.e. mu_sig expected signal counts"""
        return self._lstat_vectorized(self.mu_sig)

    @property
    def stat_max(self):
        """Stat value for best fit hypothesis.

        i.e. expected signal mu = n_on - alpha * n_off - mu_sig

        Note: For negative excess, we clip to zero since L-stat requires
        non-negative signal.
        """
        mu_best = np.clip(self.n_sig + self.mu_sig, 0, None)
        return self._lstat_vectorized(mu_best)

    def _stat_fcn(self, mu, delta=0, index=None):
        """Statistical function for root finding.

        Parameters
        ----------
        mu : float
            Signal value
        delta : float
            Offset from stat_max
        index : tuple or None
            Index for multi-dimensional arrays

        Returns
        -------
        float
            Difference between L-stat at mu and reference value
        """
        if index is None:
            mu_test = mu
        else:
            mu_test = self.mu_sig.copy()
            mu_test[index] = mu + self.mu_sig[index]

        stat_val = self._lstat_vectorized(mu_test)

        if index is not None:
            stat_val = stat_val[index]
        else:
            stat_val = stat_val

        return stat_val - delta

    def _n_sig_matching_significance_fcn(self, n_sig, significance, index):
        """Function to find excess matching given significance.

        Parameters
        ----------
        n_sig : float
            Signal counts
        significance : float
            Target significance
        index : tuple
            Index for array element

        Returns
        -------
        float
            Difference between achieved and target significance
        """
        # Stat for background-only hypothesis
        n_on_test = n_sig + self.n_bkg[index]

        stat0 = self._lstat_term(0, n_on_test, self.n_off[index], self.beta[index])

        # Stat for signal hypothesis
        stat1 = self._lstat_term(n_sig, n_on_test, self.n_off[index], self.beta[index])

        # Compute significance
        ts = np.clip(stat0 - stat1, 0, None)
        sig = np.sign(n_sig) * np.sqrt(ts)

        return sig - significance

    def sum(self, axis=None):
        """Return summed LStatCountsStatistic.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes on which to perform the summation.
            Default, axis=None, will perform the sum over the whole array.

        Returns
        -------
        stat : `LStatCountsStatistic`
            Summed statistics object
        """
        n_on = self.n_on.sum(axis=axis)
        n_off = self.n_off.sum(axis=axis)
        # Compute effective alpha from summed background
        alpha = self.n_bkg.sum(axis=axis) / n_off
        mu_sig = self.mu_sig.sum(axis=axis)

        return LStatCountsStatistic(
            n_on=n_on, n_off=n_off, alpha=alpha, mu_sig=mu_sig, tolerance=self.tolerance
        )

    def __getitem__(self, key):
        """Access statistics for a subset of bins.

        Parameters
        ----------
        key : int, slice, or tuple
            Index or slice

        Returns
        -------
        stat : `~gammapy.stats.LStatCountsStatistic`
            Statistics object for subset
        """
        return LStatCountsStatistic(
            n_on=self.n_on[key],
            n_off=self.n_off[key],
            alpha=self.alpha[key],
            mu_sig=self.mu_sig[key],
            tolerance=self.tolerance,
        )
