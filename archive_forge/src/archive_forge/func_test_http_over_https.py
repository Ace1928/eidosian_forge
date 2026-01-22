import errno
import mimetypes
import socket
import sys
from unittest import mock
import urllib.parse
from http.client import HTTPConnection
import cherrypy
from cherrypy._cpcompat import HTTPSConnection
from cherrypy.test import helper
def test_http_over_https(self):
    if self.scheme != 'https':
        return self.skip('skipped (not running HTTPS)... ')
    conn = HTTPConnection('%s:%s' % (self.interface(), self.PORT))
    conn.putrequest('GET', '/', skip_host=True)
    conn.putheader('Host', self.HOST)
    conn.endheaders()
    response = conn.response_class(conn.sock, method='GET')
    try:
        response.begin()
        self.assertEqual(response.status, 400)
        self.body = response.read()
        self.assertBody('The client sent a plain HTTP request, but this server only speaks HTTPS on this port.')
    except socket.error:
        e = sys.exc_info()[1]
        if e.errno != errno.ECONNRESET:
            raise