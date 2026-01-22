import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, DataShapeSyntaxError
def test_array_str(self):
    self.assertEqual(str(dshape('3*5*int16')), '3 * 5 * int16')