import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def std_random(self):
    return np.sqrt(np.diag(self.cov_random()))