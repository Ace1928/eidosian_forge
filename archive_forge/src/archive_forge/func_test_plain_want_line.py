from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_plain_want_line(self):
    self.assertEqual((b'want bla', []), extract_want_line_capabilities(b'want bla'))