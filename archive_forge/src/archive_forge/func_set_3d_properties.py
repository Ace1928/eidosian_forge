import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def set_3d_properties(self):
    self.update_scalarmappable()
    self._sort_zpos = None
    self.set_zsort('average')
    self._facecolor3d = PolyCollection.get_facecolor(self)
    self._edgecolor3d = PolyCollection.get_edgecolor(self)
    self._alpha3d = PolyCollection.get_alpha(self)
    self.stale = True