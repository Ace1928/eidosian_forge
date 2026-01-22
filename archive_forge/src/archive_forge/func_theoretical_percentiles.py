from statsmodels.compat.python import lzip
import numpy as np
from scipy import stats
from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import add_constant
from . import utils
@cache_readonly
def theoretical_percentiles(self):
    """Theoretical percentiles"""
    return plotting_pos(self.nobs, self.a)