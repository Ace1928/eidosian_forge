from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_var_dims(self):
    self.assertEqual(parse('var * bool', self.sym), ct.DataShape(ct.Var(), ct.bool_))
    self.assertEqual(parse('var * var * bool', self.sym), ct.DataShape(ct.Var(), ct.Var(), ct.bool_))
    self.assertEqual(parse('M * 5 * var * bool', self.sym), ct.DataShape(ct.TypeVar('M'), ct.Fixed(5), ct.Var(), ct.bool_))