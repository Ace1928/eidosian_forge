import numpy as np
from scipy.interpolate import interp1d, interp2d, Rbf
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def polyn(self):
    polyn = [interp1d(self.size, self.crit_table[:, i]) for i in range(self.n_alpha)]
    return polyn