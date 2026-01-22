import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_alpha(self, alpha):
    super().set_alpha(alpha)
    if self._gapcolor is not None:
        self.set_gapcolor(self._original_gapcolor)