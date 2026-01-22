import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
def resampled(self, lutsize):
    """Return a new colormap with *lutsize* entries."""
    colors = self(np.linspace(0, 1, lutsize))
    new_cmap = ListedColormap(colors, name=self.name)
    new_cmap._rgba_over = self._rgba_over
    new_cmap._rgba_under = self._rgba_under
    new_cmap._rgba_bad = self._rgba_bad
    return new_cmap