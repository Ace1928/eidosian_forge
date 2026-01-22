import sys
import numpy as np
from matplotlib import _api
def get_cpp_triangulation(self):
    """
        Return the underlying C++ Triangulation object, creating it
        if necessary.
        """
    from matplotlib import _tri
    if self._cpp_triangulation is None:
        self._cpp_triangulation = _tri.Triangulation(self.x, self.y, self.triangles, self.mask if self.mask is not None else (), self._edges if self._edges is not None else (), self._neighbors if self._neighbors is not None else (), not self.is_delaunay)
    return self._cpp_triangulation