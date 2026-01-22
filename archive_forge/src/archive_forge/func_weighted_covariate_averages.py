import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
@cache_readonly
def weighted_covariate_averages(self):
    """
        The average covariate values within the at-risk set at each
        event time point, weighted by hazard.
        """
    return self.model.weighted_covariate_averages(self.params)