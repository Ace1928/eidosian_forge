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
def set_xticks(self, ticks, minor=False, crs=None):
    """
        Set the x ticks.

        Parameters
        ----------
        ticks
            List of floats denoting the desired position of x ticks.
        minor: optional
            flag indicating whether the ticks should be minor
            ticks i.e. small and unlabelled (defaults to False).
        crs: optional
            An instance of :class:`~cartopy.crs.CRS` indicating the
            coordinate system of the provided tick values. If no
            coordinate system is specified then the values are assumed
            to be in the coordinate system of the projection.
            Only transformations from one rectangular coordinate system
            to another rectangular coordinate system are supported (defaults
            to None).

        Note
        ----
            This interface is subject to change whilst functionality is added
            to support other map projections.

        """
    if crs is not None and crs != self.projection:
        if not isinstance(crs, (ccrs._RectangularProjection, ccrs.Mercator)) or not isinstance(self.projection, (ccrs._RectangularProjection, ccrs.Mercator)):
            raise RuntimeError('Cannot handle non-rectangular coordinate systems.')
        proj_xyz = self.projection.transform_points(crs, np.asarray(ticks), np.zeros(len(ticks)))
        xticks = proj_xyz[..., 0]
    else:
        xticks = ticks
    self.xaxis.set_visible(True)
    return super().set_xticks(xticks, minor=minor)