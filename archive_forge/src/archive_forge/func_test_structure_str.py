import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, DataShapeSyntaxError
def test_structure_str(self):
    self.assertEqual(str(dshape('{x:int32, y:int64}')), '{x: int32, y: int64}')