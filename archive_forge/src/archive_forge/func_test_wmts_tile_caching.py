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
@pytest.mark.filterwarnings('ignore:TileMatrixLimits')
@pytest.mark.network
@pytest.mark.skipif(not _HAS_PYKDTREE_OR_SCIPY or not _OWSLIB_AVAILABLE, reason='OWSLib and at least one of pykdtree or scipy is required')
@pytest.mark.xfail(reason='NASA servers are returning bad content metadata')
def test_wmts_tile_caching():
    image_cache = WMTSRasterSource._shared_image_cache
    image_cache.clear()
    assert len(image_cache) == 0
    url = 'https://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
    wmts = WebMapTileService(url)
    layer_name = 'MODIS_Terra_CorrectedReflectance_TrueColor'
    source = WMTSRasterSource(wmts, layer_name)
    crs = ccrs.PlateCarree()
    extent = (-180, 180, -90, 90)
    resolution = (20, 10)
    n_tiles = 2
    with mock.patch.object(wmts, 'gettile', wraps=wmts.gettile) as gettile_counter:
        source.fetch_raster(crs, extent, resolution)
    assert gettile_counter.call_count == n_tiles, f'Too many tile requests - expected {n_tiles}, got {gettile_counter.call_count}.'
    del gettile_counter
    gc.collect()
    assert len(image_cache) == 1
    assert len(image_cache[wmts]) == 1
    tiles_key = (layer_name, '0')
    assert len(image_cache[wmts][tiles_key]) == n_tiles
    with mock.patch.object(wmts, 'gettile', wraps=wmts.gettile) as gettile_counter:
        source.fetch_raster(crs, extent, resolution)
    gettile_counter.assert_not_called()
    del gettile_counter
    gc.collect()
    assert len(image_cache) == 1
    assert len(image_cache[wmts]) == 1
    tiles_key = (layer_name, '0')
    assert len(image_cache[wmts][tiles_key]) == n_tiles
    del source, wmts
    gc.collect()
    assert len(image_cache) == 0