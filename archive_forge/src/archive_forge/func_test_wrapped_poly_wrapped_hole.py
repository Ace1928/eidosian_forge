import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_wrapped_poly_wrapped_hole(self):
    proj = ccrs.PlateCarree(-180)
    poly = sgeom.Polygon(ring(-40, -40, 40, 40, True), [ring(-20, -20, 20, 20, False)])
    multi_polygon = proj.project_geometry(poly)
    assert len(multi_polygon.geoms) == 2
    assert len(multi_polygon.geoms[0].interiors) == 0
    assert len(multi_polygon.geoms[1].interiors) == 0
    polygon = multi_polygon.geoms[0]
    self._assert_bounds(polygon.bounds, 140, -47, 180, 47)
    polygon = multi_polygon.geoms[1]
    self._assert_bounds(polygon.bounds, -180, -47, -140, 47)