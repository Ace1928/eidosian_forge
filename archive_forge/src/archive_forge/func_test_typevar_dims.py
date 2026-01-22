from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_typevar_dims(self):
    self.assertEqual(parse('M * bool', self.sym), ct.DataShape(ct.TypeVar('M'), ct.bool_))
    self.assertEqual(parse('A * B * bool', self.sym), ct.DataShape(ct.TypeVar('A'), ct.TypeVar('B'), ct.bool_))
    self.assertEqual(parse('A... * X * 3 * bool', self.sym), ct.DataShape(ct.Ellipsis(ct.TypeVar('A')), ct.TypeVar('X'), ct.Fixed(3), ct.bool_))