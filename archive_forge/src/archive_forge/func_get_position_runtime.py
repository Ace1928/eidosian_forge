import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
def get_position_runtime(self, ax, renderer):
    if self._locator is None:
        return self.get_position()
    else:
        return self._locator(ax, renderer).bounds