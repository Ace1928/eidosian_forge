import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_verts_and_codes(self, verts, codes):
    """Initialize vertices with path codes."""
    if len(verts) != len(codes):
        raise ValueError("'codes' must be a 1D list or array with the same length of 'verts'")
    self._paths = [mpath.Path(xy, cds) if len(xy) else mpath.Path(xy) for xy, cds in zip(verts, codes)]
    self.stale = True