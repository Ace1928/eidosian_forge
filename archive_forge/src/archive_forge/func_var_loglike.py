from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import (
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary
def var_loglike(resid, omega, nobs):
    """
    Returns the value of the VAR(p) log-likelihood.

    Parameters
    ----------
    resid : ndarray (T x K)
    omega : ndarray
        Sigma hat matrix.  Each element i,j is the average product of the
        OLS residual for variable i and the OLS residual for variable j or
        np.dot(resid.T,resid)/nobs.  There should be no correction for the
        degrees of freedom.
    nobs : int

    Returns
    -------
    llf : float
        The value of the loglikelihood function for a VAR(p) model

    Notes
    -----
    The loglikelihood function for the VAR(p) is

    .. math::

        -\\left(\\frac{T}{2}\\right)
        \\left(\\ln\\left|\\Omega\\right|-K\\ln\\left(2\\pi\\right)-K\\right)
    """
    logdet = logdet_symm(np.asarray(omega))
    neqs = len(omega)
    part1 = -(nobs * neqs / 2) * np.log(2 * np.pi)
    part2 = -(nobs / 2) * (logdet + neqs)
    return part1 + part2