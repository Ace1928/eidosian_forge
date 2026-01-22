import numpy as np
from .. import h5t, h5a
from .common import ut, TestCase
def test_key_type(self):
    with self.assertRaises(TypeError):
        self.f.attrs.create(1, data=('a', 'b'))