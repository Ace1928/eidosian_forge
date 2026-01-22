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
def error_page_namespace(k, v):
    """Attach error pages declared in config."""
    if k != 'default':
        k = int(k)
    cherrypy.serving.request.error_page[k] = v