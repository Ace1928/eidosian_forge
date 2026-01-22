import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_positions(self, positions):
    """Set the positions of the events."""
    if positions is None:
        positions = []
    if np.ndim(positions) != 1:
        raise ValueError('positions must be one-dimensional')
    lineoffset = self.get_lineoffset()
    linelength = self.get_linelength()
    pos_idx = 0 if self.is_horizontal() else 1
    segments = np.empty((len(positions), 2, 2))
    segments[:, :, pos_idx] = np.sort(positions)[:, None]
    segments[:, 0, 1 - pos_idx] = lineoffset + linelength / 2
    segments[:, 1, 1 - pos_idx] = lineoffset - linelength / 2
    self.set_segments(segments)