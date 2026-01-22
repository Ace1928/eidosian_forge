import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def get_edgecolor(self):
    ec = super().get_edgecolor()
    unmasked_polys = self._get_unmasked_polys().ravel()
    if len(ec) != len(unmasked_polys):
        return ec
    return ec[unmasked_polys, :]