import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def set_sort_zpos(self, val):
    """Set the position to use for z-sorting."""
    self._sort_zpos = val
    self.stale = True