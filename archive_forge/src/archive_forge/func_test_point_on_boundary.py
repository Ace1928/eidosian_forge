import itertools
import time
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
def test_point_on_boundary(self):
    projection = FakeProjection()
    line_string = sgeom.LineString([(180, 0), (-160, 0)])
    multi_line_string = projection.project_geometry(line_string)
    assert len(multi_line_string.geoms) == 1
    assert len(multi_line_string.geoms[0].coords) == 2
    projection = FakeProjection(left_offset=5)
    line_string = sgeom.LineString([(180, 0), (-160, 0)])
    multi_line_string = projection.project_geometry(line_string)
    assert len(multi_line_string.geoms) == 1
    assert len(multi_line_string.geoms[0].coords) == 2