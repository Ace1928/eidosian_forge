import warnings
import itertools
from contextlib import contextmanager
from packaging.version import Version
import numpy as np
import matplotlib as mpl
from matplotlib import transforms
from .. import utils
@contextmanager
def draw_figure(self, fig, props):
    if hasattr(self, '_current_fig') and self._current_fig is not None:
        warnings.warn('figure embedded in figure: something is wrong')
    self._current_fig = fig
    self._fig_props = props
    self.open_figure(fig=fig, props=props)
    yield
    self.close_figure(fig=fig)
    self._current_fig = None
    self._fig_props = {}