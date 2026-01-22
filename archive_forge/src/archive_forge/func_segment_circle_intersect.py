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
def segment_circle_intersect(x0, y0, x1, y1):
    epsilon = 1e-09
    if x1 < x0:
        x0e, x1e = (x1, x0)
    else:
        x0e, x1e = (x0, x1)
    if y1 < y0:
        y0e, y1e = (y1, y0)
    else:
        y0e, y1e = (y0, y1)
    xys = line_circle_intersect(x0, y0, x1, y1)
    xs, ys = xys.T
    return xys[(x0e - epsilon < xs) & (xs < x1e + epsilon) & (y0e - epsilon < ys) & (ys < y1e + epsilon)]