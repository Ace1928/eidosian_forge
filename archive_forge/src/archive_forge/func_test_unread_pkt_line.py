from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_unread_pkt_line(self):
    self.rin.write(b'0007foo0000')
    self.rin.seek(0)
    self.assertEqual(b'foo', self.proto.read_pkt_line())
    self.proto.unread_pkt_line(b'bar')
    self.assertEqual(b'bar', self.proto.read_pkt_line())
    self.assertEqual(None, self.proto.read_pkt_line())
    self.proto.unread_pkt_line(b'baz1')
    self.assertRaises(ValueError, self.proto.unread_pkt_line, b'baz2')