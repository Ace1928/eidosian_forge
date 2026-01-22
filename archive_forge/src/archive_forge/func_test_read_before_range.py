import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_read_before_range(self):
    f = self._file
    f._pos = 0
    self.assertRaises(errors.InvalidRange, f.read, 2)