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
def plot_ccpr_grid(results, exog_idx=None, grid=None, fig=None):
    """
    Generate CCPR plots against a set of regressors, plot in a grid.

    Generates a grid of component and component-plus-residual (CCPR) plots.

    Parameters
    ----------
    results : result instance
        A results instance with exog and params.
    exog_idx : None or list of int
        The indices or column names of the exog used in the plot.
    grid : None or tuple of int (nrows, ncols)
        If grid is given, then it is used for the arrangement of the subplots.
        If grid is None, then ncol is one, if there are only 2 subplots, and
        the number of columns is two otherwise.
    fig : Figure, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    plot_ccpr : Creates CCPR plot for a single regressor.

    Notes
    -----
    Partial residual plots are formed as::

        Res + Betahat(i)*Xi versus Xi

    and CCPR adds::

        Betahat(i)*Xi versus Xi

    References
    ----------
    See http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/ccpr.htm

    Examples
    --------
    Using the state crime dataset separately plot the effect of the each
    variable on the on the outcome, murder rate while accounting for the effect
    of all other variables in the model.

    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.formula.api as smf

    >>> fig = plt.figure(figsize=(8, 8))
    >>> crime_data = sm.datasets.statecrime.load_pandas()
    >>> results = smf.ols('murder ~ hs_grad + urban + poverty + single',
    ...                   data=crime_data.data).fit()
    >>> sm.graphics.plot_ccpr_grid(results, fig=fig)
    >>> plt.show()

    .. plot:: plots/graphics_regression_ccpr_grid.py
    """
    fig = utils.create_mpl_fig(fig)
    exog_name, exog_idx = utils.maybe_name_or_idx(exog_idx, results.model)
    if grid is not None:
        nrows, ncols = grid
    elif len(exog_idx) > 2:
        nrows = int(np.ceil(len(exog_idx) / 2.0))
        ncols = 2
    else:
        nrows = len(exog_idx)
        ncols = 1
    seen_constant = 0
    for i, idx in enumerate(exog_idx):
        if results.model.exog[:, idx].var() == 0:
            seen_constant = 1
            continue
        ax = fig.add_subplot(nrows, ncols, i + 1 - seen_constant)
        fig = plot_ccpr(results, exog_idx=idx, ax=ax)
        ax.set_title('')
    fig.suptitle('Component-Component Plus Residual Plot', fontsize='large')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    return fig