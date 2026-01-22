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
def resid_var(self):
    """estimate of variance of the residuals

        ::

           sigma2 = sigma2_OLS * (1 - hii)

        where hii is the diagonal of the hat matrix
        """
    return self.scale * (1 - self.hat_matrix_diag)