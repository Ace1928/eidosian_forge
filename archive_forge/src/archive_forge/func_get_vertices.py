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
def get_vertices(self):
    """
        Return the vertices coordinates of the ellipse.

        The definition can be found `here <https://en.wikipedia.org/wiki/Ellipse>`_

        .. versionadded:: 3.8
        """
    if self.width < self.height:
        ret = self.get_patch_transform().transform([(0, 1), (0, -1)])
    else:
        ret = self.get_patch_transform().transform([(1, 0), (-1, 0)])
    return [tuple(x) for x in ret]