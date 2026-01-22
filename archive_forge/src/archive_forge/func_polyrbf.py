import numpy as np
from scipy.interpolate import interp1d, interp2d, Rbf
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def polyrbf(self):
    xs, xa = np.meshgrid(self.size.astype(float), self.alpha)
    polyrbf = Rbf(xs.ravel(), xa.ravel(), self.crit_table.T.ravel(), function='linear')
    return polyrbf