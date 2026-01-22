import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
def test_iter_dimensionscales(self):

    def func(dsid):
        res = h5py.h5ds.get_scale_name(dsid)
        if res == b'x2 name':
            return dsid
    res = h5py.h5ds.iterate(self.f['data'].id, 2, func, 0)
    self.assertEqual(h5py.h5ds.get_scale_name(res), b'x2 name')