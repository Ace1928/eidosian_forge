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
def influence(self):
    """Influence measure

        matches the influence measure that gretl reports
        u * h / (1 - h)
        where u are the residuals and h is the diagonal of the hat_matrix
        """
    hii = self.hat_matrix_diag
    return self.resid * hii / (1 - hii)