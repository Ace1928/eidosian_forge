import os
import numpy as np
import h5py
from .common import ut, TestCase
def test_filter_ref_obj_eq():
    gzip8 = h5py.filters.Gzip(level=8)
    assert gzip8 == h5py.filters.Gzip(level=8)
    assert gzip8 != h5py.filters.Gzip(level=7)