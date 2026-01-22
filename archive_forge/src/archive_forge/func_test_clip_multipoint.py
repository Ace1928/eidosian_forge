import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_clip_multipoint(self, multi_point, mask):
    """Clipping a multipoint feature with a polygon works as expected.
        should return a geodataframe with a single multi point feature"""
    clipped = clip(multi_point, mask)
    assert clipped.geom_type[0] == 'MultiPoint'
    assert hasattr(clipped, 'attr')
    assert len(clipped) == 2
    clipped_mutltipoint = MultiPoint([Point(2, 2), Point(3, 4), Point(9, 8)])
    assert clipped.iloc[0].geometry.wkt == clipped_mutltipoint.wkt
    shape_for_points = box(*mask) if _mask_is_list_like_rectangle(mask) else mask.unary_union
    assert all(clipped.intersects(shape_for_points))