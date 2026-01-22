from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_write_pkt_line(self):
    self.proto.write_pkt_line(b'bla')
    self.assertEqual(self.rout.getvalue(), b'0007bla')