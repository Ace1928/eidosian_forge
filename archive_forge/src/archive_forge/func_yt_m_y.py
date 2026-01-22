import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
def yt_m_y(self, y):
    return np.dot(y.T, np.dot(self.m, y))