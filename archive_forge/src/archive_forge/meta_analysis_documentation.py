import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.base import HolderTuple
Forest plot with means and confidence intervals

        Parameters
        ----------
        ax : None or matplotlib axis instance
            If ax is provided, then the plot will be added to it.
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.
        use_exp : bool
            If `use_exp` is True, then the effect size and confidence limits
            will be exponentiated. This transform log-odds-ration into
            odds-ratio, and similarly for risk-ratio.
        ax : AxesSubplot, optional
            If given, this axes is used to plot in instead of a new figure
            being created.
        kwds : optional keyword arguments
            Keywords are forwarded to the dot_plot function that creates the
            plot.

        Returns
        -------
        fig : Matplotlib figure instance

        See Also
        --------
        dot_plot

        