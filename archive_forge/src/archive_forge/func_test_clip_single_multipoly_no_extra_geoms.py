import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.mark.parametrize('mask_fixture_name', mask_variants_large_rectangle)
def test_clip_single_multipoly_no_extra_geoms(buffered_locations, mask_fixture_name, request):
    """When clipping a multi-polygon feature, no additional geom types
    should be returned."""
    masks = request.getfixturevalue(mask_fixture_name)
    multi = buffered_locations.dissolve(by='type').reset_index()
    clipped = clip(multi, masks)
    assert clipped.geom_type[0] == 'Polygon'