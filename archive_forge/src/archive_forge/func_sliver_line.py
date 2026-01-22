import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.fixture
def sliver_line():
    """Create a line that will create a point when clipped."""
    linea = LineString([(10, 5), (13, 5), (15, 5)])
    lineb = LineString([(1, 1), (2, 2), (3, 2), (5, 3), (12, 1)])
    gdf = GeoDataFrame([1, 2], geometry=[linea, lineb], crs='EPSG:3857')
    return gdf