from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.graphics import utils
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import GLS, OLS, WLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tools.tools import maybe_unwrap_results
from ._regressionplots_doc import (
@Appender(_plot_partial_residuals_doc % {'extra_params_doc': 'results : object\n    Results for a fitted regression model'})
def plot_partial_residuals(results, focus_exog, ax=None):
    model = results.model
    focus_exog, focus_col = utils.maybe_name_or_idx(focus_exog, model)
    pr = partial_resids(results, focus_exog)
    focus_exog_vals = results.model.exog[:, focus_col]
    fig, ax = utils.create_mpl_ax(ax)
    ax.plot(focus_exog_vals, pr, 'o', alpha=0.6)
    ax.set_title('Partial residuals plot', fontsize='large')
    if isinstance(focus_exog, str):
        xname = focus_exog
    else:
        xname = model.exog_names[focus_exog]
    ax.set_xlabel(xname, size=15)
    ax.set_ylabel('Component plus residual', size=15)
    return fig