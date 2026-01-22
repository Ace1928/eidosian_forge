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
@cache_readonly
def dffits_internal(self):
    """dffits measure for influence of an observation

        based on resid_studentized_internal
        uses original results, no nobs loop
        """
    hii = self.hat_matrix_diag
    dffits_ = self.resid_studentized_internal * np.sqrt(hii / (1 - hii))
    dffits_threshold = 2 * np.sqrt(self.k_vars * 1.0 / self.nobs)
    return (dffits_, dffits_threshold)