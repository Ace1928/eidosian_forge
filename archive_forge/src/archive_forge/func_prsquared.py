import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families
@cache_readonly
def prsquared(self):
    """Cox-Snell Likelihood-Ratio pseudo-R-squared.

        1 - exp((llnull - .llf) * (2 / nobs))
        """
    return self.pseudo_rsquared(kind='lr')