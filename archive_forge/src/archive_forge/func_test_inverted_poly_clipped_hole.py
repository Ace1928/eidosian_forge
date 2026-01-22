import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_inverted_poly_clipped_hole(self):
    proj = ccrs.NorthPolarStereo()
    poly = sgeom.Polygon([(0, 0), (-90, 0), (-180, 0), (-270, 0)], [[(-135, -60), (-45, -60), (45, -60), (135, -60)]])
    multi_polygon = proj.project_geometry(poly)
    assert len(multi_polygon.geoms) == 1
    assert len(multi_polygon.geoms[0].interiors) == 1
    polygon = multi_polygon.geoms[0]
    self._assert_bounds(polygon.bounds, -50000000.0, -50000000.0, 50000000.0, 50000000.0, 1000000.0)
    self._assert_bounds(polygon.interiors[0].bounds, -12000000.0, -12000000.0, 12000000.0, 12000000.0, 1000000.0)
    assert abs(polygon.area - 7300000000000000.0) < 10000000000000.0