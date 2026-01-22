import numpy as np
import pytest
from shapely.geometry import MultiPoint
from shapely.geos import geos_version
from shapely.ops import voronoi_diagram
from shapely.wkt import loads as load_wkt
@requires_geos_35
def test_from_multipoint_without_tolerace_without_floating_point_coordinates():
    """But it's fine without it."""
    mp = load_wkt('MULTIPOINT (0 0, 1 0, 1 2, 0 1)')
    regions = voronoi_diagram(mp)
    assert len(regions.geoms) == 4