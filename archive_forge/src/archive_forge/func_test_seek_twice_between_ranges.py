import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_seek_twice_between_ranges(self):
    f = self._file
    start = self.first_range_start
    f.seek(start + 40)
    self.assertRaises(errors.InvalidRange, f.seek, start + 41)