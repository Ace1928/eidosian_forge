from collections.abc import MutableMapping
import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.artist import allow_rasterization
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.path as mpath
def set_patch_arc(self, center, radius, theta1, theta2):
    """Set the spine to be arc-like."""
    self._patch_type = 'arc'
    self._center = center
    self._width = radius * 2
    self._height = radius * 2
    self._theta1 = theta1
    self._theta2 = theta2
    self._path = mpath.Path.arc(theta1, theta2)
    self.set_transform(self.axes.transAxes)
    self.stale = True