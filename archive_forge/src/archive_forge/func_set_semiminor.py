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
def set_semiminor(self, b):
    """
        Set the semi-minor axis *b* of the annulus.

        Parameters
        ----------
        b : float
        """
    self.b = float(b)
    self._path = None
    self.stale = True