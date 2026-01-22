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
def getranks(self):
    """convert data to rankdata and attach


        This creates rankdata as it is used for non-parametric tests, where
        in the case of ties the average rank is assigned.


        """
    self.ranks = GroupsStats(np.column_stack([self.data, self.groupintlab]), useranks=True)
    self.rankdata = self.ranks.groupmeanfilter