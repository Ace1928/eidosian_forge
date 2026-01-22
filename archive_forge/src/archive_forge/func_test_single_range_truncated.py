import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_single_range_truncated(self):
    out = self.get_response(_single_range_response_truncated)
    self.assertRaises(errors.ShortReadvError, out.seek, out.tell() + 51)