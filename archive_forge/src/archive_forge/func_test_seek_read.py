import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_seek_read(self):
    """Test seek/read inside the range."""
    f = self._file
    start = self.first_range_start
    self.assertEqual(start, f.tell())
    cur = start
    f.seek(start + 3)
    cur += 3
    self.assertEqual(b'def', f.read(3))
    cur += len('def')
    f.seek(4, 1)
    cur += 4
    self.assertEqual(b'klmn', f.read(4))
    cur += len('klmn')
    self.assertEqual(b'', f.read(0))
    here = f.tell()
    f.seek(0, 1)
    self.assertEqual(here, f.tell())
    self.assertEqual(cur, f.tell())