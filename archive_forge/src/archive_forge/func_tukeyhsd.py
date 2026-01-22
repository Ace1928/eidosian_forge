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
def tukeyhsd(self, alpha=0.05):
    """
        Tukey's range test to compare means of all pairs of groups

        Parameters
        ----------
        alpha : float, optional
            Value of FWER at which to calculate HSD.

        Returns
        -------
        results : TukeyHSDResults instance
            A results class containing relevant data and some post-hoc
            calculations
        """
    self.groupstats = GroupsStats(np.column_stack([self.data, self.groupintlab]), useranks=False)
    gmeans = self.groupstats.groupmean
    gnobs = self.groupstats.groupnobs
    var_ = np.var(self.groupstats.groupdemean(), ddof=len(gmeans))
    res = tukeyhsd(gmeans, gnobs, var_, df=None, alpha=alpha, q_crit=None)
    resarr = np.array(lzip(self.groupsunique[res[0][0]], self.groupsunique[res[0][1]], np.round(res[2], 4), np.round(res[8], 4), np.round(res[4][:, 0], 4), np.round(res[4][:, 1], 4), res[1]), dtype=[('group1', object), ('group2', object), ('meandiff', float), ('p-adj', float), ('lower', float), ('upper', float), ('reject', np.bool_)])
    results_table = SimpleTable(resarr, headers=resarr.dtype.names)
    results_table.title = 'Multiple Comparison of Means - Tukey HSD, ' + 'FWER=%4.2f' % alpha
    return TukeyHSDResults(self, results_table, res[5], res[1], res[2], res[3], res[4], res[6], res[7], var_, res[8])