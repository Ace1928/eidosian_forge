from oslo_serialization import base64
from oslotest import base as test_base
def test_decode_as_text(self):
    self.assertEqual('text', base64.decode_as_text(b'dGV4dA=='))
    self.assertEqual('text', base64.decode_as_text('dGV4dA=='))
    self.assertEqual('e:é', base64.decode_as_text('ZTrDqQ=='))
    self.assertEqual('e:é', base64.decode_as_text('ZTrp', encoding='latin1'))