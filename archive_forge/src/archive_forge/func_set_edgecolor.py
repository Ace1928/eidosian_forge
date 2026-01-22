import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_edgecolor(self, c):
    """
        Set the edgecolor(s) of the collection.

        Parameters
        ----------
        c : color or list of colors or 'face'
            The collection edgecolor(s).  If a sequence, the patches cycle
            through it.  If 'face', match the facecolor.
        """
    if isinstance(c, str) and c.lower() in ('none', 'face'):
        c = c.lower()
    self._original_edgecolor = c
    self._set_edgecolor(c)