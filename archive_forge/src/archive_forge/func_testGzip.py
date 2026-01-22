import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
def testGzip(self):
    zbuf = io.BytesIO()
    zfile = gzip.GzipFile(mode='wb', fileobj=zbuf, compresslevel=9)
    zfile.write(b'Hello, world')
    zfile.close()
    self.getPage('/gzip/', headers=[('Accept-Encoding', 'gzip')])
    self.assertInBody(zbuf.getvalue()[:3])
    self.assertHeader('Vary', 'Accept-Encoding')
    self.assertHeader('Content-Encoding', 'gzip')
    self.getPage('/gzip/', headers=[('Accept-Encoding', 'identity')])
    self.assertHeader('Vary', 'Accept-Encoding')
    self.assertNoHeader('Content-Encoding')
    self.assertBody('Hello, world')
    self.getPage('/gzip/', headers=[('Accept-Encoding', 'gzip;q=0')])
    self.assertHeader('Vary', 'Accept-Encoding')
    self.assertNoHeader('Content-Encoding')
    self.assertBody('Hello, world')
    self.getPage('/gzip/', headers=[('Accept-Encoding', 'gzip,deflate,')])
    self.assertStatus(200)
    self.assertNotInBody('IndexError')
    self.getPage('/gzip/', headers=[('Accept-Encoding', '*;q=0')])
    self.assertStatus(406)
    self.assertNoHeader('Content-Encoding')
    self.assertErrorPage(406, 'identity, gzip')
    self.getPage('/gzip/noshow', headers=[('Accept-Encoding', 'gzip')])
    self.assertNoHeader('Content-Encoding')
    self.assertStatus(500)
    self.assertErrorPage(500, pattern='IndexError\n')
    if cherrypy.server.protocol_version == 'HTTP/1.0' or getattr(cherrypy.server, 'using_apache', False):
        self.getPage('/gzip/noshow_stream', headers=[('Accept-Encoding', 'gzip')])
        self.assertHeader('Content-Encoding', 'gzip')
        self.assertInBody('\x1f\x8b\x08\x00')
    else:
        self.assertRaises((ValueError, IncompleteRead), self.getPage, '/gzip/noshow_stream', headers=[('Accept-Encoding', 'gzip')])