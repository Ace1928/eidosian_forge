from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def proportion_confint(count, nobs, alpha: float=0.05, method='normal'):
    """
    Confidence interval for a binomial proportion

    Parameters
    ----------
    count : {int or float, array_like}
        number of successes, can be pandas Series or DataFrame. Arrays
        must contain integer values if method is "binom_test".
    nobs : {int or float, array_like}
        total number of trials.  Arrays must contain integer values if method
        is "binom_test".
    alpha : float
        Significance level, default 0.05. Must be in (0, 1)
    method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}
        default: "normal"
        method to use for confidence interval. Supported methods:

         - `normal` : asymptotic normal approximation
         - `agresti_coull` : Agresti-Coull interval
         - `beta` : Clopper-Pearson interval based on Beta distribution
         - `wilson` : Wilson Score interval
         - `jeffreys` : Jeffreys Bayesian Interval
         - `binom_test` : Numerical inversion of binom_test

    Returns
    -------
    ci_low, ci_upp : {float, ndarray, Series DataFrame}
        lower and upper confidence level with coverage (approximately) 1-alpha.
        When a pandas object is returned, then the index is taken from `count`.

    Notes
    -----
    Beta, the Clopper-Pearson exact interval has coverage at least 1-alpha,
    but is in general conservative. Most of the other methods have average
    coverage equal to 1-alpha, but will have smaller coverage in some cases.

    The "beta" and "jeffreys" interval are central, they use alpha/2 in each
    tail, and alpha is not adjusted at the boundaries. In the extreme case
    when `count` is zero or equal to `nobs`, then the coverage will be only
    1 - alpha/2 in the case of "beta".

    The confidence intervals are clipped to be in the [0, 1] interval in the
    case of "normal" and "agresti_coull".

    Method "binom_test" directly inverts the binomial test in scipy.stats.
    which has discrete steps.

    TODO: binom_test intervals raise an exception in small samples if one
       interval bound is close to zero or one.

    References
    ----------
    .. [*] https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    .. [*] Brown, Lawrence D.; Cai, T. Tony; DasGupta, Anirban (2001).
       "Interval Estimation for a Binomial Proportion", Statistical
       Science 16 (2): 101–133. doi:10.1214/ss/1009213286.
    """
    is_scalar = np.isscalar(count) and np.isscalar(nobs)
    is_pandas = isinstance(count, (pd.Series, pd.DataFrame))
    count_a = array_like(count, 'count', optional=False, ndim=None)
    nobs_a = array_like(nobs, 'nobs', optional=False, ndim=None)

    def _check(x: np.ndarray, name: str) -> np.ndarray:
        if np.issubdtype(x.dtype, np.integer):
            return x
        y = x.astype(np.int64, casting='unsafe')
        if np.any(y != x):
            raise ValueError(f'{name} must have an integral dtype. Found data with dtype {x.dtype}')
        return y
    if method == 'binom_test':
        count_a = _check(np.asarray(count_a), 'count')
        nobs_a = _check(np.asarray(nobs_a), 'count')
    q_ = count_a / nobs_a
    alpha_2 = 0.5 * alpha
    if method == 'normal':
        std_ = np.sqrt(q_ * (1 - q_) / nobs_a)
        dist = stats.norm.isf(alpha / 2.0) * std_
        ci_low = q_ - dist
        ci_upp = q_ + dist
    elif method == 'binom_test':

        def func_factory(count: int, nobs: int) -> Callable[[float], float]:
            if hasattr(stats, 'binomtest'):

                def func(qi):
                    return stats.binomtest(count, nobs, p=qi).pvalue - alpha
            else:

                def func(qi):
                    return stats.binom_test(count, nobs, p=qi) - alpha
            return func
        bcast = np.broadcast(count_a, nobs_a)
        ci_low = np.zeros(bcast.shape)
        ci_upp = np.zeros(bcast.shape)
        index = bcast.index
        for c, n in bcast:
            reverse = False
            _q = q_.flat[index]
            if c > n // 2:
                c = n - c
                reverse = True
                _q = 1 - _q
            func = func_factory(c, n)
            if c == 0:
                ci_low.flat[index] = 0.0
            else:
                lower_bnd = _bound_proportion_confint(func, _q, lower=True)
                val, _z = optimize.brentq(func, lower_bnd, _q, full_output=True)
                if func(val) > 0:
                    power = 10
                    new_lb = val - (val - lower_bnd) / 2 ** power
                    while func(new_lb) > 0 and power >= 0:
                        power -= 1
                        new_lb = val - (val - lower_bnd) / 2 ** power
                    val, _ = _bisection_search_conservative(func, new_lb, _q)
                ci_low.flat[index] = val
            if c == n:
                ci_upp.flat[index] = 1.0
            else:
                upper_bnd = _bound_proportion_confint(func, _q, lower=False)
                val, _z = optimize.brentq(func, _q, upper_bnd, full_output=True)
                if func(val) > 0:
                    power = 10
                    new_ub = val + (upper_bnd - val) / 2 ** power
                    while func(new_ub) > 0 and power >= 0:
                        power -= 1
                        new_ub = val - (upper_bnd - val) / 2 ** power
                    val, _ = _bisection_search_conservative(func, _q, new_ub)
                ci_upp.flat[index] = val
            if reverse:
                temp = ci_upp.flat[index]
                ci_upp.flat[index] = 1 - ci_low.flat[index]
                ci_low.flat[index] = 1 - temp
            index = bcast.index
    elif method == 'beta':
        ci_low = stats.beta.ppf(alpha_2, count_a, nobs_a - count_a + 1)
        ci_upp = stats.beta.isf(alpha_2, count_a + 1, nobs_a - count_a)
        if np.ndim(ci_low) > 0:
            ci_low.flat[q_.flat == 0] = 0
            ci_upp.flat[q_.flat == 1] = 1
        else:
            ci_low = 0 if q_ == 0 else ci_low
            ci_upp = 1 if q_ == 1 else ci_upp
    elif method == 'agresti_coull':
        crit = stats.norm.isf(alpha / 2.0)
        nobs_c = nobs_a + crit ** 2
        q_c = (count_a + crit ** 2 / 2.0) / nobs_c
        std_c = np.sqrt(q_c * (1.0 - q_c) / nobs_c)
        dist = crit * std_c
        ci_low = q_c - dist
        ci_upp = q_c + dist
    elif method == 'wilson':
        crit = stats.norm.isf(alpha / 2.0)
        crit2 = crit ** 2
        denom = 1 + crit2 / nobs_a
        center = (q_ + crit2 / (2 * nobs_a)) / denom
        dist = crit * np.sqrt(q_ * (1.0 - q_) / nobs_a + crit2 / (4.0 * nobs_a ** 2))
        dist /= denom
        ci_low = center - dist
        ci_upp = center + dist
    elif method[:4] == 'jeff':
        ci_low, ci_upp = stats.beta.interval(1 - alpha, count_a + 0.5, nobs_a - count_a + 0.5)
    else:
        raise NotImplementedError(f'method {method} is not available')
    if method in ['normal', 'agresti_coull']:
        ci_low = np.clip(ci_low, 0, 1)
        ci_upp = np.clip(ci_upp, 0, 1)
    if is_pandas:
        container = pd.Series if isinstance(count, pd.Series) else pd.DataFrame
        ci_low = container(ci_low, index=count.index)
        ci_upp = container(ci_upp, index=count.index)
    if is_scalar:
        return (float(ci_low), float(ci_upp))
    return (ci_low, ci_upp)