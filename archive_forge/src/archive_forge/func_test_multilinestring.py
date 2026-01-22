import numpy as np
import pytest
from shapely import LineString, MultiLineString
from shapely.errors import EmptyPartError
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase
def test_multilinestring(self):
    geom = MultiLineString([[(1.0, 2.0), (3.0, 4.0)]])
    assert isinstance(geom, MultiLineString)
    assert len(geom.geoms) == 1
    assert dump_coords(geom) == [[(1.0, 2.0), (3.0, 4.0)]]
    a = LineString([(1.0, 2.0), (3.0, 4.0)])
    ml = MultiLineString([a])
    assert len(ml.geoms) == 1
    assert dump_coords(ml) == [[(1.0, 2.0), (3.0, 4.0)]]
    ml2 = MultiLineString(ml)
    assert len(ml2.geoms) == 1
    assert dump_coords(ml2) == [[(1.0, 2.0), (3.0, 4.0)]]
    geom = MultiLineString([((0.0, 0.0), (1.0, 2.0))])
    assert isinstance(geom.geoms[0], LineString)
    assert dump_coords(geom.geoms[0]) == [(0.0, 0.0), (1.0, 2.0)]
    with pytest.raises(IndexError):
        geom.geoms[1]
    assert geom.__geo_interface__ == {'type': 'MultiLineString', 'coordinates': (((0.0, 0.0), (1.0, 2.0)),)}