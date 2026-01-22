import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_index_none(self):
    with self.assertRaises(TypeError):
        self.dset[None]