import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_indexlist_repeated(self):
    """ we forbid repeated index values """
    with self.assertRaises(TypeError):
        self.dset[[1, 1, 2]]