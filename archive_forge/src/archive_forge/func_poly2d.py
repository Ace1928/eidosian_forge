import numpy as np
from scipy.interpolate import interp1d, interp2d, Rbf
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def poly2d(self):
    poly2d = interp2d(self.size, self.alpha, self.crit_table)
    return poly2d