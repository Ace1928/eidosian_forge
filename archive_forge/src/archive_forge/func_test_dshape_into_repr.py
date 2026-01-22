from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
def test_dshape_into_repr(self):
    for ds in self.dshapes:
        self.assertEqual(eval(repr(dshape(ds))), dshape(ds))
        for dm in self.dimensions:
            d = dshape(dm + ' * ' + ds)
            self.assertEqual(eval(repr(d)), d)