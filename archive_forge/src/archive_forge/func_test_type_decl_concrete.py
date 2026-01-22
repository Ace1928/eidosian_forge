from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
@pytest.mark.xfail(reason='type decl has been removed in the new parser')
def test_type_decl_concrete(self):
    self.assertEqual(dshape('3, int32'), dshape('type X = 3, int32'))