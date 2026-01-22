import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
@cache_readonly
def score_residuals(self):
    """
        A matrix containing the score residuals.
        """
    return self.model.score_residuals(self.params)