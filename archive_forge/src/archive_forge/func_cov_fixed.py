import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def cov_fixed(self):
    """
        Approximate covariance of estimates of fixed effects.

        Just after Display (3.10) in Laird, Lange, Stram (see help(Mixed)).
        """
    return self.Sinv