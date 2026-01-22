import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.fixture
def single_rectangle_gdf_list_bounds(single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    return list(single_rectangle_gdf.total_bounds)