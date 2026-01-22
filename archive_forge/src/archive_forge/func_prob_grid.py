import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.distributions.tools import (
@cache_readonly
def prob_grid(self):
    return cdf2prob_grid(self.cdf_grid, prepend=None)