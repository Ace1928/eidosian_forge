import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def switch_orientation(self):
    """
        Switch the orientation of the event line, either from vertical to
        horizontal or vice versus.
        """
    segments = self.get_segments()
    for i, segment in enumerate(segments):
        segments[i] = np.fliplr(segment)
    self.set_segments(segments)
    self._is_horizontal = not self.is_horizontal()
    self.stale = True