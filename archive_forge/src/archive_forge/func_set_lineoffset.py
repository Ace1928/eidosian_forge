import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_lineoffset(self, lineoffset):
    """Set the offset of the lines used to mark each event."""
    if lineoffset == self.get_lineoffset():
        return
    linelength = self.get_linelength()
    segments = self.get_segments()
    pos = 1 if self.is_horizontal() else 0
    for segment in segments:
        segment[0, pos] = lineoffset + linelength / 2.0
        segment[1, pos] = lineoffset - linelength / 2.0
    self.set_segments(segments)
    self._lineoffset = lineoffset