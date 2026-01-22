from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
def test_from_numpy_fields(self):
    import numpy as np
    dt = np.dtype('i4,i8,f8')
    ds = datashape.from_numpy((), dt)
    self.assertEqual(ds.names, ['f0', 'f1', 'f2'])
    self.assertEqual(ds.types, [datashape.int32, datashape.int64, datashape.float64])