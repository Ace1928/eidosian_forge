from oslo_serialization import base64
from oslotest import base as test_base
def test_decode_as_bytes(self):
    self.assertEqual(b'text', base64.decode_as_bytes(b'dGV4dA=='))
    self.assertEqual(b'text', base64.decode_as_bytes('dGV4dA=='))