import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_no_polygon_boundary_reversal(self):
    polygon = sgeom.Polygon([(-10, 30), (10, 60), (10, 50)])
    projection = ccrs.Robinson(170.5)
    multi_polygon = projection.project_geometry(polygon)
    for polygon in multi_polygon.geoms:
        assert polygon.is_valid