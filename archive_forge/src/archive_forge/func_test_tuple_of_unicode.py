import numpy as np
from .. import h5t, h5a
from .common import ut, TestCase
def test_tuple_of_unicode(self):
    data = ('a', 'b')
    self.f.attrs.create('x', data=data)
    result = self.f.attrs['x']
    self.assertTrue(all(result == data))
    self.assertEqual(result.dtype, np.dtype('O'))
    data_as_U_array = np.array(data)
    self.assertEqual(data_as_U_array.dtype, np.dtype('U1'))
    with self.assertRaises(TypeError):
        self.f.attrs.create('y', data=data_as_U_array)