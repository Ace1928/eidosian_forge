from pandas
import numpy as np
from statsmodels.regression.linear_model import GLS, RegressionResults
@property
def inv_rwexog(self):
    """Inverse of self.rwexog"""
    if self._inv_rwexog is None:
        self._inv_rwexog = np.linalg.inv(self.rwexog)
    return self._inv_rwexog