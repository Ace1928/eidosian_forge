import numpy as np
import pytest
import shapely.geometry as sgeom
import shapely.wkt
import cartopy.crs as ccrs
def test_project_degenerate_poly(self):
    polygon = shapely.wkt.loads('POLYGON ((178.9687499944748 70.625, 179.0625 71.875, 180.9375 71.875, 179.0625 71.875, 177.1875 71.875, 178.9687499944748 70.625))')
    source = ccrs.PlateCarree()
    target = ccrs.PlateCarree()
    polygons = target.project_geometry(polygon, source)
    assert isinstance(polygons, sgeom.MultiPolygon)