import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def grid2mask(self, xi, yi):
    """Return nearest space in mask-coords from given grid-coords."""
    return (round(xi * self.x_grid2mask), round(yi * self.y_grid2mask))