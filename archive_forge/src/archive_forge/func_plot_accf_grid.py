from statsmodels.compat.pandas import deprecate_kwarg
import calendar
import numpy as np
import pandas as pd
from statsmodels.graphics import utils
from statsmodels.tools.validation import array_like
from statsmodels.tsa.stattools import acf, pacf, ccf
def plot_accf_grid(x, *, varnames=None, fig=None, lags=None, negative_lags=True, alpha=0.05, use_vlines=True, adjusted=False, fft=False, missing='none', zero=True, auto_ylims=False, bartlett_confint=False, vlines_kwargs=None, **kwargs):
    """
    Plot auto/cross-correlation grid

    Plots lags on the horizontal axis and the correlations
    on the vertical axis of each graph.

    Parameters
    ----------
    x : array_like
        2D array of time-series values: rows are observations,
        columns are variables.
    varnames: sequence of str, optional
        Variable names to use in plot titles. If ``x`` is a pandas dataframe
        and ``varnames`` is provided, it overrides the column names
        of the dataframe. If ``varnames`` is not provided and ``x`` is not
        a dataframe, variable names ``x[0]``, ``x[1]``, etc. are generated.
    fig : Matplotlib figure instance, optional
        If given, this figure is used to plot in, otherwise a new figure
        is created.
    lags : {int, array_like}, optional
        An int or array of lag values, used on horizontal axes. Uses
        ``np.arange(lags)`` when lags is an int.  If not provided,
        ``lags=np.arange(len(corr))`` is used.
    negative_lags: bool, optional
        If True, negative lags are shown on the horizontal axes of plots
        below the main diagonal.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        plotted, e.g. if alpha=.05, 95 % confidence intervals are shown.
        If None, confidence intervals are not shown on the plot.
    use_vlines : bool, optional
        If True, shows vertical lines and markers for the correlation values.
        If False, only shows markers.  The default marker is 'o'; it can
        be overridden with a ``marker`` kwarg.
    adjusted : bool
        If True, then denominators for correlations are n-k, otherwise n.
    fft : bool, optional
        If True, computes the ACF via FFT.
    missing : str, optional
        A string in ['none', 'raise', 'conservative', 'drop'] specifying how
        NaNs are to be treated.
    zero : bool, optional
        Flag indicating whether to include the 0-lag autocorrelations
        (which are always equal to 1). Default is True.
    auto_ylims : bool, optional
        If True, adjusts automatically the vertical axis limits
        to correlation values.
    bartlett_confint : bool, default False
        If True, use Bartlett's formula to calculate confidence intervals
        in auto-correlation plots. See the description of ``plot_acf`` for
        details. This argument does not affect cross-correlation plots.
    vlines_kwargs : dict, optional
        Optional dictionary of keyword arguments that are passed to vlines.
    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.

    Returns
    -------
    Figure
        If `fig` is None, the created figure.  Otherwise, `fig` is returned.
        Plots on the grid show the cross-correlation of the row variable
        with the lags of the column variable.

    See Also
    --------
    See notes and references for statsmodels.graphics.tsaplots

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm

    >>> dta = sm.datasets.macrodata.load_pandas().data
    >>> diffed = dta.diff().dropna()
    >>> sm.graphics.tsa.plot_accf_grid(diffed[["unemp", "infl"]])
    >>> plt.show()
    """
    from statsmodels.tools.data import _is_using_pandas
    array_like(x, 'x', ndim=2)
    m = x.shape[1]
    fig = utils.create_mpl_fig(fig)
    gs = fig.add_gridspec(m, m)
    if _is_using_pandas(x, None):
        varnames = varnames or list(x.columns)

        def get_var(i):
            return x.iloc[:, i]
    else:
        varnames = varnames or [f'x[{i}]' for i in range(m)]
        x = np.asarray(x)

        def get_var(i):
            return x[:, i]
    for i in range(m):
        for j in range(m):
            ax = fig.add_subplot(gs[i, j])
            if i == j:
                plot_acf(get_var(i), ax=ax, title=f'ACF({varnames[i]})', lags=lags, alpha=alpha, use_vlines=use_vlines, adjusted=adjusted, fft=fft, missing=missing, zero=zero, auto_ylims=auto_ylims, bartlett_confint=bartlett_confint, vlines_kwargs=vlines_kwargs, **kwargs)
            else:
                plot_ccf(get_var(i), get_var(j), ax=ax, title=f'CCF({varnames[i]}, {varnames[j]})', lags=lags, negative_lags=negative_lags and i > j, alpha=alpha, use_vlines=use_vlines, adjusted=adjusted, fft=fft, auto_ylims=auto_ylims, vlines_kwargs=vlines_kwargs, **kwargs)
    return fig