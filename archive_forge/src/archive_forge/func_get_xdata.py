import copy
from numbers import Integral, Number, Real
import logging
import numpy as np
import matplotlib as mpl
from . import _api, cbook, colors as mcolors, _docstring
from .artist import Artist, allow_rasterization
from .cbook import (
from .markers import MarkerStyle
from .path import Path
from .transforms import Bbox, BboxTransformTo, TransformedPath
from ._enums import JoinStyle, CapStyle
from . import _path
from .markers import (  # noqa
def get_xdata(self, orig=True):
    """
        Return the xdata.

        If *orig* is *True*, return the original data, else the
        processed data.
        """
    if orig:
        return self._xorig
    if self._invalidx:
        self.recache()
    return self._x