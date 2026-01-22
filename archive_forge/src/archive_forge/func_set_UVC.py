import math
import numpy as np
from numpy import ma
from matplotlib import _api, cbook, _docstring
import matplotlib.artist as martist
import matplotlib.collections as mcollections
from matplotlib.patches import CirclePolygon
import matplotlib.text as mtext
import matplotlib.transforms as transforms
def set_UVC(self, U, V, C=None):
    self.u = ma.masked_invalid(U, copy=True).ravel()
    self.v = ma.masked_invalid(V, copy=True).ravel()
    if len(self.flip) == 1:
        flip = np.broadcast_to(self.flip, self.u.shape)
    else:
        flip = self.flip
    if C is not None:
        c = ma.masked_invalid(C, copy=True).ravel()
        x, y, u, v, c, flip = cbook.delete_masked_points(self.x.ravel(), self.y.ravel(), self.u, self.v, c, flip.ravel())
        _check_consistent_shapes(x, y, u, v, c, flip)
    else:
        x, y, u, v, flip = cbook.delete_masked_points(self.x.ravel(), self.y.ravel(), self.u, self.v, flip.ravel())
        _check_consistent_shapes(x, y, u, v, flip)
    magnitude = np.hypot(u, v)
    flags, barbs, halves, empty = self._find_tails(magnitude, self.rounding, **self.barb_increments)
    plot_barbs = self._make_barbs(u, v, flags, barbs, halves, empty, self._length, self._pivot, self.sizes, self.fill_empty, flip)
    self.set_verts(plot_barbs)
    if C is not None:
        self.set_array(c)
    xy = np.column_stack((x, y))
    self._offsets = xy
    self.stale = True