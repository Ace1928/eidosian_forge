import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'tools.decode.on': True, 'tools.decode.default_encoding': ['utf-16']})
def extra_charset(self, *args, **kwargs):
    return ', '.join([': '.join((k, v)) for k, v in cherrypy.request.params.items()])