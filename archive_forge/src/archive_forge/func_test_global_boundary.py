import itertools
import time
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def test_global_boundary(self):
    linear_ring = sgeom.LineString([(-180, -180), (-180, 180), (180, 180), (180, -180)])
    pc = ccrs.PlateCarree()
    merc = ccrs.Mercator()
    multi_line_string = pc.project_geometry(linear_ring, merc)
    assert len(multi_line_string.geoms) > 0
    multi_line_string = merc.project_geometry(linear_ring, merc)
    assert len(multi_line_string.geoms) > 0