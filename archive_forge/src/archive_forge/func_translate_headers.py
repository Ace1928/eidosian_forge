import sys as _sys
import io
import cherrypy as _cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cperror
from cherrypy.lib import httputil
from cherrypy.lib import is_closable_iterator
def translate_headers(self, environ):
    """Translate CGI-environ header names to HTTP header names."""
    for cgiName in environ:
        if cgiName in self.headerNames:
            yield (self.headerNames[cgiName], environ[cgiName])
        elif cgiName[:5] == 'HTTP_':
            translatedHeader = cgiName[5:].replace('_', '-')
            yield (translatedHeader, environ[cgiName])