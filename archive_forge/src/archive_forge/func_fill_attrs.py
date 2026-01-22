import numpy as np
from collections.abc import MutableMapping
from .common import TestCase, ut
import h5py
from h5py import File
from h5py import h5a,  h5t
from h5py import AttributeManager
def fill_attrs(self, track_order):
    attrs = self.f.create_group('test', track_order=track_order).attrs
    for i in range(100):
        attrs[str(i)] = i
    return attrs