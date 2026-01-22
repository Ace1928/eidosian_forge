import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def get_position_3d(self):
    """Return the (x, y, z) position of the text."""
    return (self._x, self._y, self._z)