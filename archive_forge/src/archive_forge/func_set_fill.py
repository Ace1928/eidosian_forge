import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
from .bezier import (
from .path import Path
from ._enums import JoinStyle, CapStyle
def set_fill(self, b):
    """
        Set whether to fill the patch.

        Parameters
        ----------
        b : bool
        """
    self._fill = bool(b)
    self._set_facecolor(self._original_facecolor)
    self._set_edgecolor(self._original_edgecolor)
    self.stale = True