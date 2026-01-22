import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_orientation(self, orientation):
    """
        Set the orientation of the event line.

        Parameters
        ----------
        orientation : {'horizontal', 'vertical'}
        """
    is_horizontal = _api.check_getitem({'horizontal': True, 'vertical': False}, orientation=orientation)
    if is_horizontal == self.is_horizontal():
        return
    self.switch_orientation()