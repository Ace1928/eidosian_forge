import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def compute_xtwy(self):
    """
        Utility function to compute X^tWY (transposed ?) for Unit instance.
        """
    return np.dot(np.dot(self.W, self.Y), self.X)