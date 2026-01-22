import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
def test_multipart_decoding(self):
    body = ntob('\r\n'.join(['--X', 'Content-Type: text/plain;charset=utf-16', 'Content-Disposition: form-data; name="text"', '', 'ÿþa\x00b\x00\x1c c\x00', '--X', 'Content-Type: text/plain;charset=utf-16', 'Content-Disposition: form-data; name="submit"', '', 'ÿþC\x00r\x00e\x00a\x00t\x00e\x00', '--X--']))
    (self.getPage('/reqparams', method='POST', headers=[('Content-Type', 'multipart/form-data;boundary=X'), ('Content-Length', str(len(body)))], body=body),)
    self.assertBody(b'submit: Create, text: ab\xe2\x80\x9cc')