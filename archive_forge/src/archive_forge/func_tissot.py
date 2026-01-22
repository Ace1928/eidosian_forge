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
def tissot(self, rad_km=500, lons=None, lats=None, n_samples=80, **kwargs):
    """
        Add Tissot's indicatrices to the axes.

        Parameters
        ----------
        rad_km
            The radius in km of the circles to be drawn.
        lons
            A numpy.ndarray, list or tuple of longitude values that
            locate the centre of each circle. Specifying more than one
            dimension allows individual points to be drawn whereas a
            1D array produces a grid of points.
        lats
            A numpy.ndarray, list or tuple of latitude values that
            that locate the centre of each circle. See lons.
        n_samples
            Integer number of points sampled around the circumference of
            each circle.


        ``**kwargs`` are passed through to
        :class:`cartopy.feature.ShapelyFeature`.

        """
    from cartopy import geodesic
    geod = geodesic.Geodesic()
    geoms = []
    if lons is None:
        lons = np.linspace(-180, 180, 6, endpoint=False)
    else:
        lons = np.asarray(lons)
    if lats is None:
        lats = np.linspace(-80, 80, 6)
    else:
        lats = np.asarray(lats)
    if lons.ndim == 1 or lats.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)
    lons, lats = (lons.flatten(), lats.flatten())
    if lons.shape != lats.shape:
        raise ValueError('lons and lats must have the same shape.')
    for lon, lat in zip(lons, lats):
        circle = geod.circle(lon, lat, rad_km * 1000.0, n_samples=n_samples)
        geoms.append(sgeom.Polygon(circle))
    feature = cartopy.feature.ShapelyFeature(geoms, ccrs.Geodetic(), **kwargs)
    return self.add_feature(feature)