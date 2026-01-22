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
def resid_studentized(self):
    """
        Internally studentized pearson residuals

        Notes
        -----
        residuals / sqrt( scale * (1 - hii))

        where residuals are those provided to GLMInfluence which are
        pearson residuals by default, and
        hii is the diagonal of the hat matrix.
        """
    return super().resid_studentized