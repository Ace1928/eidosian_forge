import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
def resid_score_factor(self):
    """Score residual divided by sqrt of hessian factor.

        experimental, agrees with GLMInfluence for Binomial and Gaussian.
        This corresponds to considering the linear predictors as parameters
        of the model.

        Note: Nhis might have nan values if second derivative, hessian_factor,
        is positive, i.e. loglikelihood is not globally concave w.r.t. linear
        predictor. (This occured in an example for GeneralizedPoisson)
        """
    from statsmodels.genmod.generalized_linear_model import GLM
    sf = self.results.model.score_factor(self.results.params)
    hf = self.results.model.hessian_factor(self.results.params)
    if isinstance(sf, tuple):
        sf = sf[0]
    if isinstance(hf, tuple):
        hf = hf[0]
    if not isinstance(self.results.model, GLM):
        hf = -hf
    return sf / np.sqrt(hf) / np.sqrt(1 - self.hat_matrix_diag)