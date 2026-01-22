from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_write_to_boundary(self):
    self._writer.write(b'foo')
    self._writer.write(b'barba')
    self.assertOutputEquals(b'0007foo0009barba')
    self._truncate()
    self._writer.write(b'z')
    self._writer.flush()
    self.assertOutputEquals(b'0005z')