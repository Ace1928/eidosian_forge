from statsmodels.compat.python import lzip
import numpy as np
from scipy import stats
from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import add_constant
from . import utils
def qqplot_2samples(data1, data2, xlabel=None, ylabel=None, line=None, ax=None):
    """
    Q-Q Plot of two samples' quantiles.

    Can take either two `ProbPlot` instances or two array-like objects. In the
    case of the latter, both inputs will be converted to `ProbPlot` instances
    using only the default values - so use `ProbPlot` instances if
    finer-grained control of the quantile computations is required.

    Parameters
    ----------
    data1 : {array_like, ProbPlot}
        Data to plot along x axis. If the sample sizes are unequal, the longer
        series is always plotted along the x-axis.
    data2 : {array_like, ProbPlot}
        Data to plot along y axis. Does not need to have the same number of
        observations as data 1. If the sample sizes are unequal, the longer
        series is always plotted along the x-axis.
    xlabel : {None, str}
        User-provided labels for the x-axis. If None (default),
        other values are used.
    ylabel : {None, str}
        User-provided labels for the y-axis. If None (default),
        other values are used.
    line : {None, "45", "s", "r", q"}
        Options for the reference line to which the data is compared:

        - "45" - 45-degree line
        - "s" - standardized line, the expected order statistics are scaled
          by the standard deviation of the given sample and have the mean
          added to them
        - "r" - A regression line is fit
        - "q" - A line is fit through the quartiles.
        - None - by default no reference line is added to the plot.

    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    See Also
    --------
    scipy.stats.probplot

    Notes
    -----
    1) Depends on matplotlib.
    2) If `data1` and `data2` are not `ProbPlot` instances, instances will be
       created using the default parameters. Therefore, it is recommended to use
       `ProbPlot` instance if fine-grained control is needed in the computation
       of the quantiles.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from statsmodels.graphics.gofplots import qqplot_2samples
    >>> x = np.random.normal(loc=8.5, scale=2.5, size=37)
    >>> y = np.random.normal(loc=8.0, scale=3.0, size=37)
    >>> pp_x = sm.ProbPlot(x)
    >>> pp_y = sm.ProbPlot(y)
    >>> qqplot_2samples(pp_x, pp_y)
    >>> plt.show()

    .. plot:: plots/graphics_gofplots_qqplot_2samples.py

    >>> fig = qqplot_2samples(pp_x, pp_y, xlabel=None, ylabel=None,
    ...                       line=None, ax=None)
    """
    if not isinstance(data1, ProbPlot):
        data1 = ProbPlot(data1)
    if not isinstance(data2, ProbPlot):
        data2 = ProbPlot(data2)
    if data2.data.shape[0] > data1.data.shape[0]:
        fig = data1.qqplot(xlabel=ylabel, ylabel=xlabel, line=line, other=data2, ax=ax)
    else:
        fig = data2.qqplot(xlabel=ylabel, ylabel=xlabel, line=line, other=data1, ax=ax, swap=True)
    return fig