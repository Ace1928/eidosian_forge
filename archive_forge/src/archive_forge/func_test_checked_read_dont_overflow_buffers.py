import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_checked_read_dont_overflow_buffers(self):
    f = self._file
    f._discarded_buf_size = 8
    f.seek(126)
    self.assertEqual(b'AB', f.read(2))