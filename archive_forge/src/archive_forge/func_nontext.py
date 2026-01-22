import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'tools.encode.text_only': False, 'tools.encode.add_charset': True})
def nontext(self, *args, **kwargs):
    cherrypy.response.headers['Content-Type'] = 'application/binary'
    return '\x00\x01\x02\x03'