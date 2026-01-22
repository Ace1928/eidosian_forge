import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def logL(self, ML=False):
    """
        Return log-likelihood, REML by default.
        """
    logL = 0.0
    for unit in self.units:
        logL += unit.logL(a=self.a, ML=ML)
    if not ML:
        logL += np.log(L.det(self.Sinv)) / 2
    return logL