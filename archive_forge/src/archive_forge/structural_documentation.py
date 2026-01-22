from warnings import warn
import numpy as np
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import OutputWarning, SpecificationWarning
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.tsatools import lagmat
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .initialization import Initialization
from .tools import (

        Plot the estimated components of the model.

        Parameters
        ----------
        which : {'filtered', 'smoothed'}, or None, optional
            Type of state estimate to plot. Default is 'smoothed' if smoothed
            results are available otherwise 'filtered'.
        alpha : float, optional
            The confidence intervals for the components are (1 - alpha) %
        observed : bool, optional
            Whether or not to plot the observed series against
            one-step-ahead predictions.
            Default is True.
        level : bool, optional
            Whether or not to plot the level component, if applicable.
            Default is True.
        trend : bool, optional
            Whether or not to plot the trend component, if applicable.
            Default is True.
        seasonal : bool, optional
            Whether or not to plot the seasonal component, if applicable.
            Default is True.
        freq_seasonal : bool, optional
            Whether or not to plot the frequency domain seasonal component(s),
            if applicable. Default is True.
        cycle : bool, optional
            Whether or not to plot the cyclical component, if applicable.
            Default is True.
        autoregressive : bool, optional
            Whether or not to plot the autoregressive state, if applicable.
            Default is True.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        If all options are included in the model and selected, this produces
        a 6x1 plot grid with the following plots (ordered top-to-bottom):

        0. Observed series against predicted series
        1. Level
        2. Trend
        3. Seasonal
        4. Freq Seasonal
        5. Cycle
        6. Autoregressive

        Specific subplots will be removed if the component is not present in
        the estimated model or if the corresponding keyword argument is set to
        False.

        All plots contain (1 - `alpha`) %  confidence intervals.
        