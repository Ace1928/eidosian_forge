import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
def y_minv_yt(self, y):
    return np.dot(y, np.dot(self.minv, y.T))