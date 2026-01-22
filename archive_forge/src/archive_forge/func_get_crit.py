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
def get_crit(self, alpha):
    """
        get_tukeyQcrit

        currently tukey Q, add others
        """
    q_crit = get_tukeyQcrit(self.n_vals, self.df, alpha=alpha)
    return q_crit * np.ones(self.n_vals)