import numpy as np
import pytest
from shapely.geometry import MultiPoint
from shapely.geos import geos_version
from shapely.ops import voronoi_diagram
from shapely.wkt import loads as load_wkt
@requires_geos_35
def test_two_regions():
    mp = MultiPoint(points=[(0.5, 0.5), (1.0, 1.0)])
    regions = voronoi_diagram(mp)
    assert len(regions.geoms) == 2