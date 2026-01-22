import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_clip_with_polygon(single_rectangle_gdf):
    """Test clip when using a shapely object"""
    polygon = Polygon([(0, 0), (5, 12), (10, 0), (0, 0)])
    clipped = clip(single_rectangle_gdf, polygon)
    exp_poly = polygon.intersection(Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]))
    exp = GeoDataFrame([1], geometry=[exp_poly], crs='EPSG:3857')
    exp['attr2'] = 'site-boundary'
    assert_geodataframe_equal(clipped, exp)