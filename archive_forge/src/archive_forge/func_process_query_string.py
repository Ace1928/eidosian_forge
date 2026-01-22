import sys
import time
import collections
import operator
from http.cookies import SimpleCookie, CookieError
import uuid
from more_itertools import consume
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy import _cpreqbody
from cherrypy._cperror import format_exc, bare_error
from cherrypy.lib import httputil, reprconf, encoding
def process_query_string(self):
    """Parse the query string into Python structures. (Core)"""
    try:
        p = httputil.parse_query_string(self.query_string, encoding=self.query_string_encoding)
    except UnicodeDecodeError:
        raise cherrypy.HTTPError(404, 'The given query string could not be processed. Query strings for this resource must be encoded with %r.' % self.query_string_encoding)
    self.params.update(p)