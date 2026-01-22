import datetime
import functools
import logging
from numbers import Real
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits
def reset_ticks(self):
    """
        Re-initialize the major and minor Tick lists.

        Each list starts with a single fresh Tick.
        """
    try:
        del self.majorTicks
    except AttributeError:
        pass
    try:
        del self.minorTicks
    except AttributeError:
        pass
    try:
        self.set_clip_path(self.axes.patch)
    except AttributeError:
        pass