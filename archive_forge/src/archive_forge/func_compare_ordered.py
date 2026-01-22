from pystatsmodels mailinglist 20100524
from collections import namedtuple
from statsmodels.compat.python import lzip, lrange
import copy
import math
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats, interpolate
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
from statsmodels.graphics import utils
from statsmodels.tools.sm_exceptions import ValueWarning
def compare_ordered(vals, alpha):
    """simple ordered sequential comparison of means

    vals : array_like
        means or rankmeans for independent groups

    incomplete, no return, not used yet
    """
    vals = np.asarray(vals)
    alphaf = alpha
    sortind = np.argsort(vals)
    pvals = vals[sortind]
    sortrevind = sortind.argsort()
    ntests = len(vals)
    v1, v2 = np.triu_indices(ntests, 1)
    for i in range(4):
        for j in range(4, i, -1):
            print(i, j)