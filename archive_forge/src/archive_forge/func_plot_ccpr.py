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
def plot_ccpr(results, exog_idx, ax=None):
    """
    Plot CCPR against one regressor.

    Generates a component and component-plus-residual (CCPR) plot.

    Parameters
    ----------
    results : result instance
        A regression results instance.
    exog_idx : {int, str}
        Exogenous, explanatory variable. If string is given, it should
        be the variable name that you want to use, and you can use arbitrary
        translations as with a formula.
    ax : AxesSubplot, optional
        If given, it is used to plot in instead of a new figure being
        created.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    plot_ccpr_grid : Creates CCPR plot for multiple regressors in a plot grid.

    Notes
    -----
    The CCPR plot provides a way to judge the effect of one regressor on the
    response variable by taking into account the effects of the other
    independent variables. The partial residuals plot is defined as
    Residuals + B_i*X_i versus X_i. The component adds the B_i*X_i versus
    X_i to show where the fitted line would lie. Care should be taken if X_i
    is highly correlated with any of the other independent variables. If this
    is the case, the variance evident in the plot will be an underestimate of
    the true variance.

    References
    ----------
    http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/ccpr.htm

    Examples
    --------
    Using the state crime dataset plot the effect of the rate of single
    households ('single') on the murder rate while accounting for high school
    graduation rate ('hs_grad'), percentage of people in an urban area, and rate
    of poverty ('poverty').

    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.formula.api as smf

    >>> crime_data = sm.datasets.statecrime.load_pandas()
    >>> results = smf.ols('murder ~ hs_grad + urban + poverty + single',
    ...                   data=crime_data.data).fit()
    >>> sm.graphics.plot_ccpr(results, 'single')
    >>> plt.show()

    .. plot:: plots/graphics_regression_ccpr.py
    """
    fig, ax = utils.create_mpl_ax(ax)
    exog_name, exog_idx = utils.maybe_name_or_idx(exog_idx, results.model)
    results = maybe_unwrap_results(results)
    x1 = results.model.exog[:, exog_idx]
    x1beta = x1 * results.params[exog_idx]
    ax.plot(x1, x1beta + results.resid, 'o')
    from statsmodels.tools.tools import add_constant
    mod = OLS(x1beta, add_constant(x1)).fit()
    params = mod.params
    fig = abline_plot(*params, **dict(ax=ax))
    ax.set_title('Component and component plus residual plot')
    ax.set_ylabel('Residual + %s*beta_%d' % (exog_name, exog_idx))
    ax.set_xlabel('%s' % exog_name)
    return fig