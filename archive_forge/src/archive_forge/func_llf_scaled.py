from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base import _prediction_inference as pred
from statsmodels.base._prediction_inference import PredictionResultsMean
import statsmodels.base._parameter_inference as pinfer
from statsmodels.graphics._regressionplots_doc import (
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import (
from statsmodels.tools.docstring import Docstring
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import float_like
from . import families
def llf_scaled(self, scale=None):
    """
        Return the log-likelihood at the given scale, using the
        estimated scale if the provided scale is None.  In the Gaussian
        case with linear link, the concentrated log-likelihood is
        returned.
        """
    _modelfamily = self.family
    if scale is None:
        if isinstance(self.family, families.Gaussian) and isinstance(self.family.link, families.links.Power) and (self.family.link.power == 1.0):
            scale = (np.power(self._endog - self.mu, 2) * self._iweights).sum()
            scale /= self.model.wnobs
        else:
            scale = self.scale
    val = _modelfamily.loglike(self._endog, self.mu, var_weights=self._var_weights, freq_weights=self._freq_weights, scale=scale)
    return val