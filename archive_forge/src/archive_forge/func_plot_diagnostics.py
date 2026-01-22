from statsmodels.compat.pandas import is_int_index
import contextlib
import warnings
import datetime as dt
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tools.tools import pinv_extended, Bunch
from statsmodels.tools.sm_exceptions import PrecisionWarning, ValueWarning
from statsmodels.tools.numdiff import (_get_epsilon, approx_hess_cs,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
import statsmodels.base.wrapper as wrap
import statsmodels.tsa.base.prediction as pred
from statsmodels.base.data import PandasData
import statsmodels.tsa.base.tsa_model as tsbase
from .news import NewsResults
from .simulation_smoother import SimulationSmoother
from .kalman_smoother import SmootherResults
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU, MEMORY_CONSERVE
from .initialization import Initialization
from .tools import prepare_exog, concat, _safe_cond, get_impact_dates
def plot_diagnostics(self, variable=0, lags=10, fig=None, figsize=None, truncate_endog_names=24, auto_ylims=False, bartlett_confint=False, acf_kwargs=None):
    """
        Diagnostic plots for standardized residuals of one endogenous variable

        Parameters
        ----------
        variable : int, optional
            Index of the endogenous variable for which the diagnostic plots
            should be created. Default is 0.
        lags : int, optional
            Number of lags to include in the correlogram. Default is 10.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the 2x2 grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).
        auto_ylims : bool, optional
            If True, adjusts automatically the y-axis limits to ACF values.
        bartlett_confint : bool, default True
            Confidence intervals for ACF values are generally placed at 2
            standard errors around r_k. The formula used for standard error
            depends upon the situation. If the autocorrelations are being used
            to test for randomness of residuals as part of the ARIMA routine,
            the standard errors are determined assuming the residuals are white
            noise. The approximate formula for any lag is that standard error
            of each r_k = 1/sqrt(N). See section 9.4 of [1] for more details on
            the 1/sqrt(N) result. For more elementary discussion, see section
            5.3.2 in [2].
            For the ACF of raw data, the standard error at a lag k is
            found as if the right model was an MA(k-1). This allows the
            possible interpretation that if all autocorrelations past a
            certain lag are within the limits, the model might be an MA of
            order defined by the last significant autocorrelation. In this
            case, a moving average model is assumed for the data and the
            standard errors for the confidence intervals should be
            generated using Bartlett's formula. For more details on
            Bartlett formula result, see section 7.2 in [1].+
        acf_kwargs : dict, optional
            Optional dictionary of keyword arguments that are directly passed
            on to the correlogram Matplotlib plot produced by plot_acf().

        Returns
        -------
        Figure
            Figure instance with diagnostic plots

        See Also
        --------
        statsmodels.graphics.gofplots.qqplot
        statsmodels.graphics.tsaplots.plot_acf

        Notes
        -----
        Produces a 2x2 plot grid with the following plots (ordered clockwise
        from top left):

        1. Standardized residuals over time
        2. Histogram plus estimated density of standardized residuals, along
           with a Normal(0,1) density plotted for reference.
        3. Normal Q-Q plot, with Normal reference line.
        4. Correlogram

        References
        ----------
        [1] Brockwell and Davis, 1987. Time Series Theory and Methods
        [2] Brockwell and Davis, 2010. Introduction to Time Series and
        Forecasting, 2nd edition.
        """
    from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
    _import_mpl()
    fig = create_mpl_fig(fig, figsize)
    d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
    if isinstance(variable, str):
        variable = self.model.endog_names.index(variable)
    if hasattr(self.data, 'dates') and self.data.dates is not None:
        ix = self.data.dates[d:]
    else:
        ix = np.arange(self.nobs - d)
    resid = pd.Series(self.filter_results.standardized_forecasts_error[variable, d:], index=ix)
    if resid.shape[0] < max(d, lags):
        raise ValueError('Length of endogenous variable must be larger the the number of lags used in the model and the number of observations burned in the log-likelihood calculation.')
    ax = fig.add_subplot(221)
    resid.dropna().plot(ax=ax)
    ax.hlines(0, ix[0], ix[-1], alpha=0.5)
    ax.set_xlim(ix[0], ix[-1])
    name = self.model.endog_names[variable]
    if len(name) > truncate_endog_names:
        name = name[:truncate_endog_names - 3] + '...'
    ax.set_title(f'Standardized residual for "{name}"')
    resid_nonmissing = resid.dropna()
    ax = fig.add_subplot(222)
    ax.hist(resid_nonmissing, density=True, label='Hist', edgecolor='#FFFFFF')
    from scipy.stats import gaussian_kde, norm
    kde = gaussian_kde(resid_nonmissing)
    xlim = (-1.96 * 2, 1.96 * 2)
    x = np.linspace(xlim[0], xlim[1])
    ax.plot(x, kde(x), label='KDE')
    ax.plot(x, norm.pdf(x), label='N(0,1)')
    ax.set_xlim(xlim)
    ax.legend()
    ax.set_title('Histogram plus estimated density')
    ax = fig.add_subplot(223)
    from statsmodels.graphics.gofplots import qqplot
    qqplot(resid_nonmissing, line='s', ax=ax)
    ax.set_title('Normal Q-Q')
    ax = fig.add_subplot(224)
    from statsmodels.graphics.tsaplots import plot_acf
    if acf_kwargs is None:
        acf_kwargs = {}
    plot_acf(resid, ax=ax, lags=lags, auto_ylims=auto_ylims, bartlett_confint=bartlett_confint, **acf_kwargs)
    ax.set_title('Correlogram')
    return fig