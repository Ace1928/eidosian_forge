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
def runbasic(self, useranks=False):
    """runbasic"""
    x = self.x
    if useranks:
        xuni, xintlab = np.unique(x[:, 0], return_inverse=True)
        ranksraw = x[:, 0].argsort().argsort() + 1
        self.xx = GroupsStats(np.column_stack([ranksraw, xintlab]), useranks=False).groupmeanfilter
    else:
        self.xx = x[:, 0]
    self.groupsum = groupranksum = np.bincount(self.intlab, weights=self.xx)
    self.groupmean = grouprankmean = groupranksum * 1.0 / self.groupnobs
    self.groupmeanfilter = grouprankmean[self.intlab]