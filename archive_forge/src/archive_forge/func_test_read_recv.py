from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_read_recv(self):
    all_data = b'12345678abcdefg'
    self.rin.write(all_data)
    self.rin.seek(0)
    self.assertEqual(b'1234', self.proto.read(4))
    self.assertEqual(b'5678abc', self.proto.recv(8))
    self.assertEqual(b'defg', self.proto.read(4))
    self.assertRaises(GitProtocolError, self.proto.recv, 10)