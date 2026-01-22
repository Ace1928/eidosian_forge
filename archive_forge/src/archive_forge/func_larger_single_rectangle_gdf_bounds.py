import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.fixture
def larger_single_rectangle_gdf_bounds(larger_single_rectangle_gdf):
    """Bounds of the created single rectangle"""
    return tuple(larger_single_rectangle_gdf.total_bounds)