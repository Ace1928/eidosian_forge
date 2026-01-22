import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
@property
def bse(self):
    """
        standard errors of estimated coefficients for exogeneous variables (fixed)

        """
    return np.sqrt(np.diag(self.cov_params()))