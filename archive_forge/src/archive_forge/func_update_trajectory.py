import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def update_trajectory(self, xg, yg, broken_streamlines=True):
    if not self.grid.within_grid(xg, yg):
        raise InvalidIndexError
    xm, ym = self.grid2mask(xg, yg)
    self.mask._update_trajectory(xm, ym, broken_streamlines)