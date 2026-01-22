import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_out_of_bounds(self):
    projection = ccrs.TransverseMercator(central_longitude=0, approx=True)
    polys = [([(86, -1), (86, 1), (88, 1), (88, -1)], 1), ([(86, -1), (86, 1), (130, 1), (88, -1)], 1), ([(86, -1), (86, 1), (130, 1), (130, -1)], 1), ([(120, -1), (120, 1), (130, 1), (130, -1)], 0)]
    for coords, expected_polys in polys:
        polygon = sgeom.Polygon(coords)
        multi_polygon = projection.project_geometry(polygon)
        assert len(multi_polygon.geoms) == expected_polys