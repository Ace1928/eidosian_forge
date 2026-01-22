from oslo_serialization import base64
from oslotest import base as test_base
def test_decode_as_bytes__error(self):
    self.assertRaises(TypeError, base64.decode_as_bytes, 'hello world')