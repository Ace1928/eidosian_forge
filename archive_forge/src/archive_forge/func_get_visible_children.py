import itertools
import operator
import warnings
import matplotlib
import matplotlib.artist
import matplotlib.collections as mcollections
import matplotlib.text
import matplotlib.ticker as mticker
import matplotlib.transforms as mtrans
import numpy as np
import shapely.geometry as sgeom
import cartopy
from cartopy.crs import PlateCarree, Projection, _RectangularProjection
from cartopy.mpl.ticker import (
def get_visible_children(self):
    """Return a list of the visible child `.Artist`\\s."""
    all_children = self.xline_artists + self.yline_artists + self.label_artists
    return [c for c in all_children if c.get_visible()]