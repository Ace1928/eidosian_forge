import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def determine_facecolor(patch):
    if patch.get_fill():
        return patch.get_facecolor()
    return [0, 0, 0, 0]