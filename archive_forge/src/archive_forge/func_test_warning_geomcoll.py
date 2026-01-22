import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_warning_geomcoll(self, geomcol_gdf, mask):
    """Test the correct warnings are raised if keep_geom_type is
        called on a GDF with GeometryCollection"""
    with pytest.warns(UserWarning):
        clip(geomcol_gdf, mask, keep_geom_type=True)