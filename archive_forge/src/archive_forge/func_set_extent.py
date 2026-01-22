import collections
import contextlib
import functools
import json
import os
from pathlib import Path
import warnings
import weakref
import matplotlib as mpl
import matplotlib.artist
import matplotlib.axes
import matplotlib.contour
from matplotlib.image import imread
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.ma as ma
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl import _MPL_38
import cartopy.mpl.contour
import cartopy.mpl.feature_artist as feature_artist
import cartopy.mpl.geocollection
import cartopy.mpl.patch as cpatch
from cartopy.mpl.slippy_image_artist import SlippyImageArtist
def set_extent(self, extents, crs=None):
    """
        Set the extent (x0, x1, y0, y1) of the map in the given
        coordinate system.

        If no crs is given, the extents' coordinate system will be assumed
        to be the Geodetic version of this axes' projection.

        Parameters
        ----------
        extents
            Tuple of floats representing the required extent (x0, x1, y0, y1).
        """
    x1, x2, y1, y2 = extents
    domain_in_crs = sgeom.polygon.LineString([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
    projected = None
    try_workaround = crs is None and isinstance(self.projection, ccrs.PlateCarree) or crs == self.projection
    if try_workaround:
        boundary = self.projection.boundary
        if boundary.equals(domain_in_crs):
            projected = boundary
    if projected is None:
        projected = self.projection.project_geometry(domain_in_crs, crs)
    try:
        x1, y1, x2, y2 = projected.bounds
    except ValueError:
        raise ValueError(f'Failed to determine the required bounds in projection coordinates. Check that the values provided are within the valid range (x_limits={self.projection.x_limits}, y_limits={self.projection.y_limits}).')
    self.set_xlim([x1, x2])
    self.set_ylim([y1, y2])