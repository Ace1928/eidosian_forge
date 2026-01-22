import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_antialiased(self, aa):
    """
        Set the antialiasing state for rendering.

        Parameters
        ----------
        aa : bool or list of bools
        """
    if aa is None:
        aa = self._get_default_antialiased()
    self._antialiaseds = np.atleast_1d(np.asarray(aa, bool))
    self.stale = True