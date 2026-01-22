from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_recv(self):
    all_data = b'1234567' * 10
    self.rin.write(all_data)
    self.rin.seek(0)
    data = b''
    for _ in range(10):
        data += self.proto.recv(10)
    self.assertRaises(GitProtocolError, self.proto.recv, 10)
    self.assertEqual(all_data, data)