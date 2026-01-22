from statsmodels.compat.python import asbytes
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
def get_thsd(mci, alpha=0.05):
    var_ = np.var(mci.groupstats.groupdemean(), ddof=len(mci.groupsunique))
    means = mci.groupstats.groupmean
    nobs = mci.groupstats.groupnobs
    resi = tukeyhsd(means, nobs, var_, df=None, alpha=alpha, q_crit=qsturng(1 - alpha, len(means), (nobs - 1).sum()))
    var2 = (mci.groupstats.groupvarwithin() * (nobs - 1.0)).sum() / (nobs - 1.0).sum()
    assert_almost_equal(var_, var2, decimal=14)
    return resi