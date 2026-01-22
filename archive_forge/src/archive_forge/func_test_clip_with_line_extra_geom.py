import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_clip_with_line_extra_geom(self, sliver_line, mask):
    """When the output of a clipped line returns a geom collection,
        and keep_geom_type is True, no geometry collections should be returned."""
    clipped = clip(sliver_line, mask, keep_geom_type=True)
    assert len(clipped.geometry) == 1
    assert not (clipped.geom_type == 'GeometryCollection').any()