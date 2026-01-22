import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.mpl.feature_artist import FeatureArtist, _freeze, _GeomKey
def cached_paths(geom, target_projection):
    geom_cache = FeatureArtist._geom_key_to_path_cache.get(_GeomKey(geom), {})
    return geom_cache.get(target_projection, None)