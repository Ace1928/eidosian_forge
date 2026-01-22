import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def compute_xtwx(self):
    """
        Utility function to compute X^tWX for Unit instance.
        """
    return np.dot(np.dot(self.X.T, self.W), self.X)