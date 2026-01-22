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
def get_minorticklabels(self):
    """Return this Axis' minor tick labels, as a list of `~.text.Text`."""
    self._update_ticks()
    ticks = self.get_minor_ticks()
    labels1 = [tick.label1 for tick in ticks if tick.label1.get_visible()]
    labels2 = [tick.label2 for tick in ticks if tick.label2.get_visible()]
    return labels1 + labels2