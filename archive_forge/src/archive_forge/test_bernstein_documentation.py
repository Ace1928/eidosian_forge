import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from scipy import stats
from statsmodels.distributions.copula.api import (
from statsmodels.distributions.copula.api import transforms as tra
import statsmodels.distributions.tools as dt
from statsmodels.distributions.bernstein import (

Created on Wed Feb 17 23:44:18 2021

Author: Josef Perktold
License: BSD-3

