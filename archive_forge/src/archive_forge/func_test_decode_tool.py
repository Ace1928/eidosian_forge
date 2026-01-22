import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
def test_decode_tool(self):
    body = b'\xff\xfeq\x00=\xff\xfe\xa3\x00'
    (self.getPage('/decode/extra_charset', method='POST', headers=[('Content-Type', 'application/x-www-form-urlencoded'), ('Content-Length', str(len(body)))], body=body),)
    self.assertBody(b'q: \xc2\xa3')
    body = b'q=\xc2\xa3'
    (self.getPage('/decode/extra_charset', method='POST', headers=[('Content-Type', 'application/x-www-form-urlencoded'), ('Content-Length', str(len(body)))], body=body),)
    self.assertBody(b'q: \xc2\xa3')
    body = b'q=\xc2\xa3'
    (self.getPage('/decode/force_charset', method='POST', headers=[('Content-Type', 'application/x-www-form-urlencoded'), ('Content-Length', str(len(body)))], body=body),)
    self.assertErrorPage(400, "The request entity could not be decoded. The following charsets were attempted: ['utf-16']")