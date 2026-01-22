import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
def test_delete_replies_202(self):
    """A bogus return code for delete raises an error."""
    self.server.canned_response = b'HTTP/1.1 202 OK\r\nDate: Tue, 10 Aug 2013 14:38:56 GMT\r\nServer: Apache/42 (Wonderland)\r\n'
    t = self.get_transport()
    self.assertRaises(errors.InvalidHttpResponse, t.delete, 'whatever')