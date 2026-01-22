import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_seek_across_ranges(self):
    f = self._file
    f.seek(126)
    self.assertEqual(b'AB', f.read(2))