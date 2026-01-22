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
def plot_regress_exog(results, exog_idx, fig=None):
    """Plot regression results against one regressor.

    This plots four graphs in a 2 by 2 figure: 'endog versus exog',
    'residuals versus exog', 'fitted versus exog' and
    'fitted plus residual versus exog'

    Parameters
    ----------
    results : result instance
        A result instance with resid, model.endog and model.exog as attributes.
    exog_idx : int or str
        Name or index of regressor in exog matrix.
    fig : Figure, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    Figure
        The value of `fig` if provided. Otherwise a new instance.

    Examples
    --------
    Load the Statewide Crime data set and build a model with regressors
    including the rate of high school graduation (hs_grad), population in urban
    areas (urban), households below poverty line (poverty), and single person
    households (single).  Outcome variable is the murder rate (murder).

    Build a 2 by 2 figure based on poverty showing fitted versus actual murder
    rate, residuals versus the poverty rate, partial regression plot of poverty,
    and CCPR plot for poverty rate.

    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.formula.api as smf

    >>> fig = plt.figure(figsize=(8, 6))
    >>> crime_data = sm.datasets.statecrime.load_pandas()
    >>> results = smf.ols('murder ~ hs_grad + urban + poverty + single',
    ...                   data=crime_data.data).fit()
    >>> sm.graphics.plot_regress_exog(results, 'poverty', fig=fig)
    >>> plt.show()

    .. plot:: plots/graphics_regression_regress_exog.py
    """
    fig = utils.create_mpl_fig(fig)
    exog_name, exog_idx = utils.maybe_name_or_idx(exog_idx, results.model)
    results = maybe_unwrap_results(results)
    y_name = results.model.endog_names
    x1 = results.model.exog[:, exog_idx]
    prstd, iv_l, iv_u = wls_prediction_std(results)
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x1, results.model.endog, 'o', color='b', alpha=0.9, label=y_name)
    ax.plot(x1, results.fittedvalues, 'D', color='r', label='fitted', alpha=0.5)
    ax.vlines(x1, iv_l, iv_u, linewidth=1, color='k', alpha=0.7)
    ax.set_title('Y and Fitted vs. X', fontsize='large')
    ax.set_xlabel(exog_name)
    ax.set_ylabel(y_name)
    ax.legend(loc='best')
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x1, results.resid, 'o')
    ax.axhline(y=0, color='black')
    ax.set_title('Residuals versus %s' % exog_name, fontsize='large')
    ax.set_xlabel(exog_name)
    ax.set_ylabel('resid')
    ax = fig.add_subplot(2, 2, 3)
    exog_noti = np.ones(results.model.exog.shape[1], bool)
    exog_noti[exog_idx] = False
    exog_others = results.model.exog[:, exog_noti]
    from pandas import Series
    fig = plot_partregress(results.model.data.orig_endog, Series(x1, name=exog_name, index=results.model.data.row_labels), exog_others, obs_labels=False, ax=ax)
    ax.set_title('Partial regression plot', fontsize='large')
    ax = fig.add_subplot(2, 2, 4)
    fig = plot_ccpr(results, exog_idx, ax=ax)
    ax.set_title('CCPR Plot', fontsize='large')
    fig.suptitle('Regression Plots for %s' % exog_name, fontsize='large')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    return fig