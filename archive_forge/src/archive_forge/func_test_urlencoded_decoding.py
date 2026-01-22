import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
def test_urlencoded_decoding(self):
    europoundUtf8 = europoundUnicode.encode('utf-8')
    body = b'param=' + europoundUtf8
    (self.getPage('/', method='POST', headers=[('Content-Type', 'application/x-www-form-urlencoded'), ('Content-Length', str(len(body)))], body=body),)
    self.assertBody(europoundUtf8)
    body = b'q=\xc2\xa3'
    (self.getPage('/reqparams', method='POST', headers=[('Content-Type', 'application/x-www-form-urlencoded'), ('Content-Length', str(len(body)))], body=body),)
    self.assertBody(b'q: \xc2\xa3')
    body = b'\xff\xfeq\x00=\xff\xfe\xa3\x00'
    (self.getPage('/reqparams', method='POST', headers=[('Content-Type', 'application/x-www-form-urlencoded;charset=utf-16'), ('Content-Length', str(len(body)))], body=body),)
    self.assertBody(b'q: \xc2\xa3')
    body = b'\xff\xfeq\x00=\xff\xfe\xa3\x00'
    (self.getPage('/reqparams', method='POST', headers=[('Content-Type', 'application/x-www-form-urlencoded;charset=utf-8'), ('Content-Length', str(len(body)))], body=body),)
    self.assertStatus(400)
    self.assertErrorPage(400, "The request entity could not be decoded. The following charsets were attempted: ['utf-8']")