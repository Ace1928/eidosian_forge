from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_unsigned_integers(self):
    self.assertEqual(parse('uint8', self.sym), ct.DataShape(ct.uint8))
    self.assertEqual(parse('uint16', self.sym), ct.DataShape(ct.uint16))
    self.assertEqual(parse('uint32', self.sym), ct.DataShape(ct.uint32))
    self.assertEqual(parse('uint64', self.sym), ct.DataShape(ct.uint64))
    self.assertEqual(parse('uintptr', self.sym), ct.DataShape(ct.uintptr))