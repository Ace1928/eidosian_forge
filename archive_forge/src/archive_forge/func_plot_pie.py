from __future__ import annotations
import collections
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
@add_fig_kwargs
def plot_pie(self, key='wall_time', minfract=0.05, **kwargs):
    """
        Plot pie charts of the different timers.

        Args:
            key: Keyword used to extract data from timers.
            minfract: Don't show sections whose relative weight is less that minfract.

        Returns:
            `matplotlib` figure
        """
    timers = self.timers()
    n = len(timers)
    fig = plt.gcf()
    gspec = GridSpec(n, 1)
    for idx, timer in enumerate(timers):
        ax = plt.subplot(gspec[idx, 0])
        ax.set_title(str(timer))
        timer.pie(ax=ax, key=key, minfract=minfract, show=False)
    return fig