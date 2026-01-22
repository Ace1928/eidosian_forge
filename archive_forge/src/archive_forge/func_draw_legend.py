import warnings
import itertools
from contextlib import contextmanager
from packaging.version import Version
import numpy as np
import matplotlib as mpl
from matplotlib import transforms
from .. import utils
@contextmanager
def draw_legend(self, legend, props):
    self._current_legend = legend
    self._legend_props = props
    self.open_legend(legend=legend, props=props)
    yield
    self.close_legend(legend=legend)
    self._current_legend = None
    self._legend_props = {}