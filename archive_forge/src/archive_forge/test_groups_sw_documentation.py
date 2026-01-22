import numpy as np
from numpy.testing import assert_equal, assert_raises
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.grouputils import GroupSorted
Test for a helper function for PanelHAC robust covariance

the functions should be rewritten to make it more efficient

Created on Thu May 17 21:09:41 2012

Author: Josef Perktold
