import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.fixture
def point_gdf():
    """Create a point GeoDataFrame."""
    pts = np.array([[2, 2], [3, 4], [9, 8], [-12, -15]])
    gdf = GeoDataFrame([Point(xy) for xy in pts], columns=['geometry'], crs='EPSG:3857')
    return gdf