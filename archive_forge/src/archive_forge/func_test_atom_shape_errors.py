from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
def test_atom_shape_errors(self):
    self.assertRaises(error.DataShapeSyntaxError, dshape, 'boot')
    self.assertRaises(error.DataShapeSyntaxError, dshape, 'int33')
    self.assertRaises(error.DataShapeSyntaxError, dshape, '12')
    self.assertRaises(error.DataShapeSyntaxError, dshape, 'var')