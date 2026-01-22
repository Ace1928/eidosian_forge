import functools
import logging
import math
from numbers import Real
import weakref
import numpy as np
import matplotlib as mpl
from . import _api, artist, cbook, _docstring
from .artist import Artist
from .font_manager import FontProperties
from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from .textpath import TextPath, TextToPath  # noqa # Logically located here
from .transforms import (
def get_unitless_position(self):
    """Return the (x, y) unitless position of the text."""
    x = float(self.convert_xunits(self._x))
    y = float(self.convert_yunits(self._y))
    return (x, y)