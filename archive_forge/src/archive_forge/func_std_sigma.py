import numpy as np
from scipy import special
from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf
@property
def std_sigma(self):
    """standard deviation, square root of diagonal elements of sigma
        """
    return np.sqrt(np.diag(self.sigma))