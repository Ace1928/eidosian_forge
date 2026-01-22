import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_non_overlapping_geoms():
    """Test that a bounding box returns empty if the extents don't overlap"""
    unit_box = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    unit_gdf = GeoDataFrame([1], geometry=[unit_box], crs='EPSG:3857')
    non_overlapping_gdf = unit_gdf.copy()
    non_overlapping_gdf = non_overlapping_gdf.geometry.apply(lambda x: shapely.affinity.translate(x, xoff=20))
    out = clip(unit_gdf, non_overlapping_gdf)
    assert_geodataframe_equal(out, unit_gdf.iloc[:0])
    out2 = clip(unit_gdf.geometry, non_overlapping_gdf)
    assert_geoseries_equal(out2, GeoSeries(crs=unit_gdf.crs))