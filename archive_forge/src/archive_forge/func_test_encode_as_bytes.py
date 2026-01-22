from oslo_serialization import base64
from oslotest import base as test_base
def test_encode_as_bytes(self):
    self.assertEqual(b'dGV4dA==', base64.encode_as_bytes(b'text'))
    self.assertEqual(b'dGV4dA==', base64.encode_as_bytes('text'))
    self.assertEqual(b'ZTrDqQ==', base64.encode_as_bytes('e:é'))
    self.assertEqual(b'ZTrp', base64.encode_as_bytes('e:é', encoding='latin1'))