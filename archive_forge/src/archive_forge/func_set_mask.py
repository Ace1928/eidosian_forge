import sys
import numpy as np
from matplotlib import _api
def set_mask(self, mask):
    """
        Set or clear the mask array.

        Parameters
        ----------
        mask : None or bool array of length ntri
        """
    if mask is None:
        self.mask = None
    else:
        self.mask = np.asarray(mask, dtype=bool)
        if self.mask.shape != (self.triangles.shape[0],):
            raise ValueError('mask array must have same length as triangles array')
    if self._cpp_triangulation is not None:
        self._cpp_triangulation.set_mask(self.mask if self.mask is not None else ())
    self._edges = None
    self._neighbors = None
    if self._trifinder is not None:
        self._trifinder._initialize()