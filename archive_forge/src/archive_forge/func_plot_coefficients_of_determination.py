from statsmodels.compat.pandas import MONTH_END, QUARTER_END
from collections import OrderedDict
from warnings import warn
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import int_like
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import string_like
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.base.data import PandasData
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.tableformatting import fmt_params
def plot_coefficients_of_determination(self, method='individual', which=None, endog_labels=None, fig=None, figsize=None):
    """
        Plot coefficients of determination (R-squared) for variables / factors.

        Parameters
        ----------
        method : {'individual', 'joint', 'cumulative'}, optional
            The type of R-squared values to generate. "individual" plots
            the R-squared of each variable on each factor; "joint" plots the
            R-squared of each variable on each factor that it loads on;
            "cumulative" plots the successive R-squared values as each
            additional factor is added to the regression, for each variable.
            Default is 'individual'.
        which: {None, 'filtered', 'smoothed'}, optional
            Whether to compute R-squared values based on filtered or smoothed
            estimates of the factors. Default is 'smoothed' if smoothed results
            are available and 'filtered' otherwise.
        endog_labels : bool, optional
            Whether or not to label the endogenous variables along the x-axis
            of the plots. Default is to include labels if there are 5 or fewer
            endogenous variables.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        The endogenous variables are arranged along the x-axis according to
        their position in the model's `endog` array.

        See Also
        --------
        get_coefficients_of_determination
        """
    from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
    _import_mpl()
    fig = create_mpl_fig(fig, figsize)
    method = string_like(method, 'method', options=['individual', 'joint', 'cumulative'])
    if endog_labels is None:
        endog_labels = self.model.k_endog <= 5
    rsquared = self.get_coefficients_of_determination(method=method, which=which)
    if method in ['individual', 'cumulative']:
        plot_idx = 1
        for factor_name, coeffs in rsquared.T.iterrows():
            ax = fig.add_subplot(self.model.k_factors, 1, plot_idx)
            ax.set_ylim((0, 1))
            ax.set(title=f'{factor_name}', ylabel='$R^2$')
            coeffs.plot(ax=ax, kind='bar')
            if plot_idx < len(rsquared.columns) or not endog_labels:
                ax.xaxis.set_ticklabels([])
            plot_idx += 1
    elif method == 'joint':
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylim((0, 1))
        ax.set(title='$R^2$ - regression on all loaded factors', ylabel='$R^2$')
        rsquared.plot(ax=ax, kind='bar')
        if not endog_labels:
            ax.xaxis.set_ticklabels([])
    return fig