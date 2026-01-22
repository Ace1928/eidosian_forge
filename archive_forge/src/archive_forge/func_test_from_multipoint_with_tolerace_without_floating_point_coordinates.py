import numpy as np
import pytest
from shapely.geometry import MultiPoint
from shapely.geos import geos_version
from shapely.ops import voronoi_diagram
from shapely.wkt import loads as load_wkt
@requires_geos_35
def test_from_multipoint_with_tolerace_without_floating_point_coordinates():
    """This multipoint will not work with a tolerance value."""
    mp = load_wkt('MULTIPOINT (0 0, 1 0, 1 2, 0 1)')
    with pytest.raises(ValueError) as exc:
        voronoi_diagram(mp, tolerance=0.1)
    assert 'Could not create Voronoi Diagram with the specified inputs' in str(exc.value)
    assert 'Try running again with default tolerance value.' in str(exc.value)