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
def power_ztost_prop(low, upp, nobs, p_alt, alpha=0.05, dist='norm', variance_prop=None, discrete=True, continuity=0, critval_continuity=0):
    """
    Power of proportions equivalence test based on normal distribution

    Parameters
    ----------
    low, upp : floats
        lower and upper limit of equivalence region
    nobs : int
        number of observations
    p_alt : float in (0,1)
        proportion under the alternative
    alpha : float in (0,1)
        significance level of the test
    dist : str in ['norm', 'binom']
        This defines the distribution to evaluate the power of the test. The
        critical values of the TOST test are always based on the normal
        approximation, but the distribution for the power can be either the
        normal (default) or the binomial (exact) distribution.
    variance_prop : None or float in (0,1)
        If this is None, then the variances for the two one sided tests are
        based on the proportions equal to the equivalence limits.
        If variance_prop is given, then it is used to calculate the variance
        for the TOST statistics. If this is based on an sample, then the
        estimated proportion can be used.
    discrete : bool
        If true, then the critical values of the rejection region are converted
        to integers. If dist is "binom", this is automatically assumed.
        If discrete is false, then the TOST critical values are used as
        floating point numbers, and the power is calculated based on the
        rejection region that is not discretized.
    continuity : bool or float
        adjust the rejection region for the normal power probability. This has
        and effect only if ``dist='norm'``
    critval_continuity : bool or float
        If this is non-zero, then the critical values of the tost rejection
        region are adjusted before converting to integers. This affects both
        distributions, ``dist='norm'`` and ``dist='binom'``.

    Returns
    -------
    power : float
        statistical power of the equivalence test.
    (k_low, k_upp, z_low, z_upp) : tuple of floats
        critical limits in intermediate steps
        temporary return, will be changed

    Notes
    -----
    In small samples the power for the ``discrete`` version, has a sawtooth
    pattern as a function of the number of observations. As a consequence,
    small changes in the number of observations or in the normal approximation
    can have a large effect on the power.

    ``continuity`` and ``critval_continuity`` are added to match some results
    of PASS, and are mainly to investigate the sensitivity of the ztost power
    to small changes in the rejection region. From my interpretation of the
    equations in the SAS manual, both are zero in SAS.

    works vectorized

    **verification:**

    The ``dist='binom'`` results match PASS,
    The ``dist='norm'`` results look reasonable, but no benchmark is available.

    References
    ----------
    SAS Manual: Chapter 68: The Power Procedure, Computational Resources
    PASS Chapter 110: Equivalence Tests for One Proportion.

    """
    mean_low = low
    var_low = std_prop(low, nobs) ** 2
    mean_upp = upp
    var_upp = std_prop(upp, nobs) ** 2
    mean_alt = p_alt
    var_alt = std_prop(p_alt, nobs) ** 2
    if variance_prop is not None:
        var_low = var_upp = std_prop(variance_prop, nobs) ** 2
    power = _power_ztost(mean_low, var_low, mean_upp, var_upp, mean_alt, var_alt, alpha=alpha, discrete=discrete, dist=dist, nobs=nobs, continuity=continuity, critval_continuity=critval_continuity)
    return (np.maximum(power[0], 0), power[1:])