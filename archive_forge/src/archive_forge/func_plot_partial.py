from collections.abc import Iterable
import copy  # check if needed when dropping python 2.7
import numpy as np
from scipy import optimize
import pandas as pd
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import (
import statsmodels.regression.linear_model as lm
from statsmodels.tools.sm_exceptions import (PerfectSeparationError,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
def plot_partial(self, smooth_index, plot_se=True, cpr=False, include_constant=True, ax=None):
    """plot the contribution of a smooth term to the linear prediction

        Parameters
        ----------
        smooth_index : int
            index of the smooth term within list of smooth terms
        plot_se : bool
            If plot_se is true, then the confidence interval for the linear
            prediction will be added to the plot.
        cpr : bool
            If cpr (component plus residual) is true, then a scatter plot of
            the partial working residuals will be added to the plot.
        include_constant : bool
            If true, then the estimated intercept is added to the prediction
            and its standard errors. This avoids that the confidence interval
            has zero width at the imposed identification constraint, e.g.
            either at a reference point or at the mean.
        ax : None or matplotlib axis instance
           If ax is not None, then the plot will be added to it.

        Returns
        -------
        Figure
            If `ax` is None, the created figure. Otherwise, the Figure to which
            `ax` is connected.
        """
    from statsmodels.graphics.utils import _import_mpl, create_mpl_ax
    _import_mpl()
    variable = smooth_index
    y_est, se = self.partial_values(variable, include_constant=include_constant)
    smoother = self.model.smoother
    x = smoother.smoothers[variable].x
    sort_index = np.argsort(x)
    x = x[sort_index]
    y_est = y_est[sort_index]
    se = se[sort_index]
    fig, ax = create_mpl_ax(ax)
    if cpr:
        residual = self.resid_working[sort_index]
        cpr_ = y_est + residual
        ax.scatter(x, cpr_, s=4)
    ax.plot(x, y_est, c='blue', lw=2)
    if plot_se:
        ax.plot(x, y_est + 1.96 * se, '-', c='blue')
        ax.plot(x, y_est - 1.96 * se, '-', c='blue')
    ax.set_xlabel(smoother.smoothers[variable].variable_name)
    return fig