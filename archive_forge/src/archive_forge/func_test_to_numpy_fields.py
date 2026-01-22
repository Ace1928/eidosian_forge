from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
def test_to_numpy_fields(self):
    import numpy as np
    ds = datashape.dshape('{x: int32, y: float32}')
    shape, dt = datashape.to_numpy(ds)
    self.assertEqual(shape, ())
    self.assertEqual(dt, np.dtype([('x', 'int32'), ('y', 'float32')]))