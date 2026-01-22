import io
import os
import sys
import re
import platform
import tempfile
import urllib.parse
import unittest.mock
from http.client import HTTPConnection
import pytest
import py.path
import path
import cherrypy
from cherrypy.lib import static
from cherrypy._cpcompat import HTTPSConnection, ntou, tonative
from cherrypy.test import helper
@pytest.mark.xfail(reason='#1475')
def test_file_stream(self):
    if cherrypy.server.protocol_version != 'HTTP/1.1':
        return self.skip()
    self.PROTOCOL = 'HTTP/1.1'
    self.persistent = True
    conn = self.HTTP_CONN
    conn.putrequest('GET', '/bigfile', skip_host=True)
    conn.putheader('Host', self.HOST)
    conn.endheaders()
    response = conn.response_class(conn.sock, method='GET')
    response.begin()
    self.assertEqual(response.status, 200)
    body = b''
    remaining = BIGFILE_SIZE
    while remaining > 0:
        data = response.fp.read(65536)
        if not data:
            break
        body += data
        remaining -= len(data)
        if self.scheme == 'https':
            newconn = HTTPSConnection
        else:
            newconn = HTTPConnection
        s, h, b = helper.webtest.openURL(b'/tell', headers=[], host=self.HOST, port=self.PORT, http_conn=newconn)
        if not b:
            tell_position = BIGFILE_SIZE
        else:
            tell_position = int(b)
        read_so_far = len(body)
        if tell_position >= BIGFILE_SIZE:
            if read_so_far < BIGFILE_SIZE / 2:
                self.fail('The file should have advanced to position %r, but has already advanced to the end of the file. It may not be streamed as intended, or at the wrong chunk size (64k)' % read_so_far)
        elif tell_position < read_so_far:
            self.fail('The file should have advanced to position %r, but has only advanced to position %r. It may not be streamed as intended, or at the wrong chunk size (64k)' % (read_so_far, tell_position))
    if body != b'x' * BIGFILE_SIZE:
        self.fail("Body != 'x' * %d. Got %r instead (%d bytes)." % (BIGFILE_SIZE, body[:50], len(body)))
    conn.close()