import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
def test_unkown_format_response(self):
    example = '<document/>'
    self.assertRaises(errors.InvalidHttpResponse, self._extract_dir_content_from_str, example)