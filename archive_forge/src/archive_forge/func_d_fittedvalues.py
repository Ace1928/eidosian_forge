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
def d_fittedvalues(self):
    """Change in expected response, fittedvalues.

        Local change of expected mean given the change in the parameters as
        computed in d_params.

        Notes
        -----
        This uses the one-step approximation of the parameter change to
        deleting one observation ``d_params``.
        """
    params = np.asarray(self.results.params)
    deriv = self.results.model._deriv_mean_dparams(params)
    return (deriv * self.d_params).sum(1)