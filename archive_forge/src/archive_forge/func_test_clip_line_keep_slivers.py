import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_clip_line_keep_slivers(sliver_line, single_rectangle_gdf):
    """Test the correct output if a point is returned
    from a line only geometry type."""
    clipped = clip(sliver_line, single_rectangle_gdf)
    assert 'Point' == clipped.geom_type[0]
    assert 'LineString' == clipped.geom_type[1]