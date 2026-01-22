import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_seek_from_end(self):
    """See TestRangeFileMixin.test_seek_from_end."""
    f = self._file
    f.seek(-2, 2)
    self.assertEqual(b'yz', f.read())
    self.assertRaises(errors.InvalidRange, f.seek, -2, 2)