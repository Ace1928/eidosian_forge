from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_send_cmd(self):
    self.proto.send_cmd(b'fetch', b'a', b'b')
    self.assertEqual(self.rout.getvalue(), b'000efetch a\x00b\x00')