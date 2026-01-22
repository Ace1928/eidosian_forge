from statsmodels.compat.python import lzip
import numpy as np
from scipy import stats
from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import add_constant
from . import utils
def qqplot(self, xlabel=None, ylabel=None, line=None, other=None, ax=None, swap: bool=False, **plotkwargs):
    """
        Plot of the quantiles of x versus the quantiles/ppf of a distribution.

        Can also be used to plot against the quantiles of another `ProbPlot`
        instance.

        Parameters
        ----------
        xlabel : {None, str}
            User-provided labels for the x-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        ylabel : {None, str}
            User-provided labels for the y-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        line : {None, "45", "s", "r", q"}, optional
            Options for the reference line to which the data is compared:

            - "45" - 45-degree line
            - "s" - standardized line, the expected order statistics are scaled
              by the standard deviation of the given sample and have the mean
              added to them
            - "r" - A regression line is fit
            - "q" - A line is fit through the quartiles.
            - None - by default no reference line is added to the plot.

        other : {ProbPlot, array_like, None}, optional
            If provided, the sample quantiles of this `ProbPlot` instance are
            plotted against the sample quantiles of the `other` `ProbPlot`
            instance. Sample size of `other` must be equal or larger than
            this `ProbPlot` instance. If the sample size is larger, sample
            quantiles of `other` will be interpolated to match the sample size
            of this `ProbPlot` instance. If an array-like object is provided,
            it will be turned into a `ProbPlot` instance using default
            parameters. If not provided (default), the theoretical quantiles
            are used.
        ax : AxesSubplot, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.
        swap : bool, optional
            Flag indicating to swap the x and y labels.
        **plotkwargs
            Additional arguments to be passed to the `plot` command.

        Returns
        -------
        Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        """
    if other is not None:
        check_other = isinstance(other, ProbPlot)
        if not check_other:
            other = ProbPlot(other)
        s_self = self.sample_quantiles
        s_other = other.sample_quantiles
        if len(s_self) > len(s_other):
            raise ValueError('Sample size of `other` must be equal or ' + 'larger than this `ProbPlot` instance')
        elif len(s_self) < len(s_other):
            p = plotting_pos(self.nobs, self.a)
            s_other = stats.mstats.mquantiles(s_other, p)
        fig, ax = _do_plot(s_other, s_self, self.dist, ax=ax, line=line, **plotkwargs)
        if xlabel is None:
            xlabel = 'Quantiles of 2nd Sample'
        if ylabel is None:
            ylabel = 'Quantiles of 1st Sample'
        if swap:
            xlabel, ylabel = (ylabel, xlabel)
    else:
        fig, ax = _do_plot(self.theoretical_quantiles, self.sample_quantiles, self.dist, ax=ax, line=line, **plotkwargs)
        if xlabel is None:
            xlabel = 'Theoretical Quantiles'
        if ylabel is None:
            ylabel = 'Sample Quantiles'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig