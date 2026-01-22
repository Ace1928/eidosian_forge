import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def set_pane_color(self, color, alpha=None):
    """
        Set pane color.

        Parameters
        ----------
        color : color
            Color for axis pane.
        alpha : float, optional
            Alpha value for axis pane. If None, base it on *color*.
        """
    color = mcolors.to_rgba(color, alpha)
    self._axinfo['color'] = color
    self.pane.set_edgecolor(color)
    self.pane.set_facecolor(color)
    self.pane.set_alpha(color[-1])
    self.stale = True