import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.axes import Axes
import matplotlib.axis as maxis
from matplotlib.patches import Circle
from matplotlib.path import Path
import matplotlib.spines as mspines
from matplotlib.ticker import (
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform
def set_longitude_grid_ends(self, degrees):
    """
        Set the latitude(s) at which to stop drawing the longitude grids.
        """
    self._longitude_cap = np.deg2rad(degrees)
    self._xaxis_pretransform.clear().scale(1.0, self._longitude_cap * 2.0).translate(0.0, -self._longitude_cap)