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
def plot_all(self, show=True, **kwargs):
    """Call all plot methods provided by the parser."""
    figs = []
    app = figs.append
    app(self.plot_stacked_hist(show=show))
    app(self.plot_efficiency(show=show))
    app(self.plot_pie(show=show))
    return figs