import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_mixed_geom(self, mixed_gdf, mask):
    """Test clipping a mixed GeoDataFrame"""
    clipped = clip(mixed_gdf, mask)
    assert clipped.geom_type[0] == 'Point' and clipped.geom_type[1] == 'Polygon' and (clipped.geom_type[2] == 'LineString')