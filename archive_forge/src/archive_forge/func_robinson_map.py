import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.mpl.feature_artist import FeatureArtist, _freeze, _GeomKey
def robinson_map():
    """
    Set up a common map for the image tests.  The extent is chosen to include
    only the square geometry from `feature`.  This means that we can check that
    `array` or a list of facecolors remains 1-to-1 with the list of geometries.
    """
    prj_crs = ccrs.Robinson()
    fig, ax = plt.subplots(subplot_kw={'projection': prj_crs})
    ax.set_extent([20, 180, -90, 90])
    ax.coastlines()
    return (fig, ax)