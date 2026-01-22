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
def set_xy(self, xy):
    """
        Set the vertices of the polygon.

        Parameters
        ----------
        xy : (N, 2) array-like
            The coordinates of the vertices.

        Notes
        -----
        Unlike `.Path`, we do not ignore the last input vertex. If the
        polygon is meant to be closed, and the last point of the polygon is not
        equal to the first, we assume that the user has not explicitly passed a
        ``CLOSEPOLY`` vertex, and add it ourselves.
        """
    xy = np.asarray(xy)
    nverts, _ = xy.shape
    if self._closed:
        if nverts == 1 or (nverts > 1 and (xy[0] != xy[-1]).any()):
            xy = np.concatenate([xy, [xy[0]]])
    elif nverts > 2 and (xy[0] == xy[-1]).all():
        xy = xy[:-1]
    self._path = Path(xy, closed=self._closed)
    self.stale = True