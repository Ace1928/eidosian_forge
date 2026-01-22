import gc
from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.io.shapereader
from cartopy.mpl import _MPL_38
from cartopy.mpl.feature_artist import FeatureArtist
import cartopy.mpl.geoaxes as cgeoaxes
import cartopy.mpl.patch
@pytest.mark.natural_earth
def test_shapefile_transform_cache():
    coastline_path = cartopy.io.shapereader.natural_earth(resolution='110m', category='physical', name='coastline')
    geoms = cartopy.io.shapereader.Reader(coastline_path).geometries()
    geoms = tuple(geoms)[:10]
    n_geom = len(geoms)
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Robinson())
    FeatureArtist._geom_key_to_geometry_cache.clear()
    FeatureArtist._geom_key_to_path_cache.clear()
    assert len(FeatureArtist._geom_key_to_geometry_cache) == 0
    assert len(FeatureArtist._geom_key_to_path_cache) == 0
    with mock.patch.object(ax.projection, 'project_geometry', wraps=ax.projection.project_geometry) as counter:
        ax.add_geometries(geoms, ccrs.PlateCarree())
        ax.add_geometries(geoms, ccrs.PlateCarree())
        ax.add_geometries(geoms[:], ccrs.PlateCarree())
        fig.canvas.draw()
    assert counter.call_count == n_geom, f'The given geometry was transformed too many times (expected: {n_geom}; got {counter.call_count}) - the caching is not working.'
    assert len(FeatureArtist._geom_key_to_geometry_cache) == n_geom
    assert len(FeatureArtist._geom_key_to_path_cache) == n_geom
    fig.clf()
    del geoms, counter
    gc.collect()
    assert len(FeatureArtist._geom_key_to_geometry_cache) == 0
    assert len(FeatureArtist._geom_key_to_path_cache) == 0