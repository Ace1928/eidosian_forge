import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.fixture
def multi_point(point_gdf):
    """Create a multi-point GeoDataFrame."""
    multi_point = point_gdf.unary_union
    out_df = GeoDataFrame(geometry=GeoSeries([multi_point, Point(2, 5), Point(-11, -14), Point(-10, -12)]), crs='EPSG:3857')
    out_df['attr'] = ['tree', 'another tree', 'shrub', 'berries']
    return out_df