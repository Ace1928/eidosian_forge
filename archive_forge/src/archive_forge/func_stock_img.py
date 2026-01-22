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
def stock_img(self, name='ne_shaded', **kwargs):
    """
        Add a standard image to the map.

        Currently, the only (and default) option for image is a downsampled
        version of the Natural Earth shaded relief raster. Other options
        (e.g., alpha) will be passed to :func:`GeoAxes.imshow`.

        """
    if name == 'ne_shaded':
        source_proj = ccrs.PlateCarree()
        fname = config['repo_data_dir'] / 'raster' / 'natural_earth' / '50-natural-earth-1-downsampled.png'
        return self.imshow(imread(fname), origin='upper', transform=source_proj, extent=[-180, 180, -90, 90], **kwargs)
    else:
        raise ValueError('Unknown stock image %r.' % name)