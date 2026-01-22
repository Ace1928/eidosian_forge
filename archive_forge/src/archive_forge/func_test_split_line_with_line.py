import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
def test_split_line_with_line(self):
    splitter = LineString([(0, 1), (1, 0)])
    self.helper(self.ls, splitter, 2)
    splitter = LineString([(0, 1), (1, 0), (1, 2)])
    self.helper(self.ls, splitter, 3)
    splitter = LineString([(0, 0), (15, 15)])
    with pytest.raises(ValueError):
        self.helper(self.ls, splitter, 1)
    splitter = LineString([(0, 1), (0, 2)])
    self.helper(self.ls, splitter, 1)
    splitter = LineString([(-1, 1), (1, -1)])
    assert splitter.touches(self.ls)
    self.helper(self.ls, splitter, 1)
    splitter = LineString([(0, 1), (1, 1)])
    assert splitter.touches(self.ls)
    self.helper(self.ls, splitter, 2)