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
def plot_partregress(endog, exog_i, exog_others, data=None, title_kwargs={}, obs_labels=True, label_kwargs={}, ax=None, ret_coords=False, eval_env=1, **kwargs):
    """Plot partial regression for a single regressor.

    Parameters
    ----------
    endog : {ndarray, str}
       The endogenous or response variable. If string is given, you can use a
       arbitrary translations as with a formula.
    exog_i : {ndarray, str}
        The exogenous, explanatory variable. If string is given, you can use a
        arbitrary translations as with a formula.
    exog_others : {ndarray, list[str]}
        Any other exogenous, explanatory variables. If a list of strings is
        given, each item is a term in formula. You can use a arbitrary
        translations as with a formula. The effect of these variables will be
        removed by OLS regression.
    data : {DataFrame, dict}
        Some kind of data structure with names if the other variables are
        given as strings.
    title_kwargs : dict
        Keyword arguments to pass on for the title. The key to control the
        fonts is fontdict.
    obs_labels : {bool, array_like}
        Whether or not to annotate the plot points with their observation
        labels. If obs_labels is a boolean, the point labels will try to do
        the right thing. First it will try to use the index of data, then
        fall back to the index of exog_i. Alternatively, you may give an
        array-like object corresponding to the observation numbers.
    label_kwargs : dict
        Keyword arguments that control annotate for the observation labels.
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    ret_coords : bool
        If True will return the coordinates of the points in the plot. You
        can use this to add your own annotations.
    eval_env : int
        Patsy eval environment if user functions and formulas are used in
        defining endog or exog.
    **kwargs
        The keyword arguments passed to plot for the points.

    Returns
    -------
    fig : Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.
    coords : list, optional
        If ret_coords is True, return a tuple of arrays (x_coords, y_coords).

    See Also
    --------
    plot_partregress_grid : Plot partial regression for a set of regressors.

    Notes
    -----
    The slope of the fitted line is the that of `exog_i` in the full
    multiple regression. The individual points can be used to assess the
    influence of points on the estimated coefficient.

    Examples
    --------
    Load the Statewide Crime data set and plot partial regression of the rate
    of high school graduation (hs_grad) on the murder rate(murder).

    The effects of the percent of the population living in urban areas (urban),
    below the poverty line (poverty) , and in a single person household (single)
    are removed by OLS regression.

    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt

    >>> crime_data = sm.datasets.statecrime.load_pandas()
    >>> sm.graphics.plot_partregress(endog='murder', exog_i='hs_grad',
    ...                              exog_others=['urban', 'poverty', 'single'],
    ...                              data=crime_data.data, obs_labels=False)
    >>> plt.show()

    .. plot:: plots/graphics_regression_partregress.py

    More detailed examples can be found in the Regression Plots notebook
    on the examples page.
    """
    fig, ax = utils.create_mpl_ax(ax)
    if isinstance(endog, str):
        endog = dmatrix(endog + '-1', data, eval_env=eval_env)
    if isinstance(exog_others, str):
        RHS = dmatrix(exog_others, data, eval_env=eval_env)
    elif isinstance(exog_others, list):
        RHS = '+'.join(exog_others)
        RHS = dmatrix(RHS, data, eval_env=eval_env)
    else:
        RHS = exog_others
    RHS_isemtpy = False
    if isinstance(RHS, np.ndarray) and RHS.size == 0:
        RHS_isemtpy = True
    elif isinstance(RHS, pd.DataFrame) and RHS.empty:
        RHS_isemtpy = True
    if isinstance(exog_i, str):
        exog_i = dmatrix(exog_i + '-1', data, eval_env=eval_env)
    if RHS_isemtpy:
        endog = np.asarray(endog)
        exog_i = np.asarray(exog_i)
        ax.plot(endog, exog_i, 'o', **kwargs)
        fitted_line = OLS(endog, exog_i).fit()
        x_axis_endog_name = 'x' if isinstance(exog_i, np.ndarray) else exog_i.name
        y_axis_endog_name = 'y' if isinstance(endog, np.ndarray) else endog.design_info.column_names[0]
    else:
        res_yaxis = OLS(endog, RHS).fit()
        res_xaxis = OLS(exog_i, RHS).fit()
        xaxis_resid = res_xaxis.resid
        yaxis_resid = res_yaxis.resid
        x_axis_endog_name = res_xaxis.model.endog_names
        y_axis_endog_name = res_yaxis.model.endog_names
        ax.plot(xaxis_resid, yaxis_resid, 'o', **kwargs)
        fitted_line = OLS(yaxis_resid, xaxis_resid).fit()
    fig = abline_plot(0, np.asarray(fitted_line.params)[0], color='k', ax=ax)
    if x_axis_endog_name == 'y':
        x_axis_endog_name = 'x'
    ax.set_xlabel('e(%s | X)' % x_axis_endog_name)
    ax.set_ylabel('e(%s | X)' % y_axis_endog_name)
    ax.set_title('Partial Regression Plot', **title_kwargs)
    if obs_labels is True:
        if data is not None:
            obs_labels = data.index
        elif hasattr(exog_i, 'index'):
            obs_labels = exog_i.index
        else:
            obs_labels = res_xaxis.model.data.row_labels
        if obs_labels is None:
            obs_labels = lrange(len(exog_i))
    if obs_labels is not False:
        if len(obs_labels) != len(exog_i):
            raise ValueError('obs_labels does not match length of exog_i')
        label_kwargs.update(dict(ha='center', va='bottom'))
        ax = utils.annotate_axes(lrange(len(obs_labels)), obs_labels, lzip(res_xaxis.resid, res_yaxis.resid), [(0, 5)] * len(obs_labels), 'x-large', ax=ax, **label_kwargs)
    if ret_coords:
        return (fig, (res_xaxis.resid, res_yaxis.resid))
    else:
        return fig