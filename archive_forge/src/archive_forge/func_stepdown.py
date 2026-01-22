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
def stepdown(self, indices):
    """stepdown"""
    print(indices)
    if self.check_set(indices):
        if len(indices) > 2:
            for subs in self.iter_subsets(indices):
                self.stepdown(subs)
        else:
            self.rejected.append(tuple(indices))
    else:
        self.accepted.append(tuple(indices))
        return indices