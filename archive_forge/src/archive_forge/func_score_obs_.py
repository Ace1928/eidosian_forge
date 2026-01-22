import warnings
from statsmodels.compat.pandas import Appender
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import stats
from statsmodels.base.model import (
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
def score_obs_(self, params):
    """score, first derivative of loglike for each observations

        This currently only implements the derivative with respect to the
        exog parameters, but not with respect to threshold parameters.

        """
    low, upp = self._bounds(params)
    prob = self.prob(low, upp)
    pdf_upp = self.pdf(upp)
    pdf_low = self.pdf(low)
    score_factor = (pdf_upp - pdf_low)[:, None]
    score_factor /= prob[:, None]
    so = np.column_stack((-score_factor[:, :1] * self.exog, score_factor[:, 1:]))
    return so