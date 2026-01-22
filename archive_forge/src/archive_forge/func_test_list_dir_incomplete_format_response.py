import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
def test_list_dir_incomplete_format_response(self):
    example = '<?xml version="1.0" encoding="utf-8"?>\n<D:multistatus xmlns:D="DAV:" xmlns:ns0="urn:uuid:c2f41010-65b3-11d1-a29f-00aa00c14882/">\n<D:response>\n<D:href>http://localhost/</D:href>\n</D:response>\n<D:response>\n<D:href>http://localhost/titi</D:href>\n</D:response>\n<D:href>http://localhost/toto</D:href>\n</D:multistatus>'
    self.assertRaises(errors.NotADirectory, self._extract_dir_content_from_str, example)