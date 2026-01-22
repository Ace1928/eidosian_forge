import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_cleanup_pipe(self):
    sock = ReadSocket(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=UTF-8\r\nContent-Length: 18\n\r\n0123456789\ngarbage')
    conn = FakeHTTPConnection(sock)
    conn.putrequest('GET', 'http://localhost/fictious')
    conn.endheaders()
    resp = conn.getresponse()
    self.assertEqual(b'0123456789\n', resp.read(11))
    conn._range_warning_thresold = 6
    conn.cleanup_pipe()
    self.assertContainsRe(self.get_log(), 'Got a 200 response when asking')