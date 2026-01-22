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
@_api.deprecated('3.8')
def set_label1(self, s):
    """
        Set the label1 text.

        Parameters
        ----------
        s : str
        """
    self.label1.set_text(s)
    self.stale = True