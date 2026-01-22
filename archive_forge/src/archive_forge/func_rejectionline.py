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
def rejectionline(n, alpha=0.5):
    """reference line for rejection in multiple tests

    Not used anymore

    from: section 3.2, page 60
    """
    t = np.arange(n) / float(n)
    frej = t / (t * (1 - alpha) + alpha)
    return frej