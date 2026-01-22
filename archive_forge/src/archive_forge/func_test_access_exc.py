import numpy as np
from collections.abc import MutableMapping
from .common import TestCase, ut
import h5py
from h5py import File
from h5py import h5a,  h5t
from h5py import AttributeManager
def test_access_exc(self):
    """ Attempt to access missing item raises KeyError """
    with self.assertRaises(KeyError):
        self.f.attrs['a']