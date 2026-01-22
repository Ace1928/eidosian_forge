import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_iter_empty(self):
    f = response.ResponseFile('empty', BytesIO())
    self.assertEqual([], list(f))