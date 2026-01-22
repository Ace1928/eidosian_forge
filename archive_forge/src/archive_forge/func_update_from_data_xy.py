import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
def update_from_data_xy(self, xy, ignore=None, updatex=True, updatey=True):
    """
        Update the `Bbox` bounds based on the passed in *xy* coordinates.

        After updating, the bounds will have positive *width* and *height*;
        *x0* and *y0* will be the minimal values.

        Parameters
        ----------
        xy : (N, 2) array-like
            The (x, y) coordinates.
        ignore : bool, optional
            - When ``True``, ignore the existing bounds of the `Bbox`.
            - When ``False``, include the existing bounds of the `Bbox`.
            - When ``None``, use the last value passed to :meth:`ignore`.
        updatex, updatey : bool, default: True
             When ``True``, update the x/y values.
        """
    if len(xy) == 0:
        return
    path = Path(xy)
    self.update_from_path(path, ignore=ignore, updatex=updatex, updatey=updatey)