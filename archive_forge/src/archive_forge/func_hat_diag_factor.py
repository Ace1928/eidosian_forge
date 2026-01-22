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
def hat_diag_factor(self):
    """Factor of diagonal of hat_matrix used in influence

        this might be useful for internal reuse
        h / (1 - h)
        """
    hii = self.hat_matrix_diag
    return hii / (1 - hii)