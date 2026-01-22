from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_read_pkt_line_none(self):
    self.rin.write(b'0000')
    self.rin.seek(0)
    self.assertEqual(None, self.proto.read_pkt_line())