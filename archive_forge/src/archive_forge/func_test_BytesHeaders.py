import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
def test_BytesHeaders(self):
    self.getPage('/cookies_and_headers')
    self.assertBody('Any content')
    self.assertHeader('Bytes-Header', 'Bytes given header')