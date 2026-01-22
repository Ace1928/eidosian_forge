import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_clip_with_multipolygon(buffered_locations, single_rectangle_gdf):
    """Test clipping a polygon with a multipolygon."""
    multi = buffered_locations.dissolve(by='type').reset_index()
    clipped = clip(single_rectangle_gdf, multi)
    assert clipped.geom_type[0] == 'Polygon'