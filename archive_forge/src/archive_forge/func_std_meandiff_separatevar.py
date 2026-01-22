import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def std_meandiff_separatevar(self):
    d1 = self.d1
    d2 = self.d2
    return np.sqrt(d1._var / (d1.nobs - 1) + d2._var / (d2.nobs - 1))