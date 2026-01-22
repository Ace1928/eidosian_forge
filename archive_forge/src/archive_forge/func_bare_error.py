import io
import contextlib
import urllib.parse
from sys import exc_info as _exc_info
from traceback import format_exception as _format_exception
from xml.sax import saxutils
import html
from more_itertools import always_iterable
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy._cpcompat import tonative
from cherrypy._helper import classproperty
from cherrypy.lib import httputil as _httputil
def bare_error(extrabody=None):
    """Produce status, headers, body for a critical error.

    Returns a triple without calling any other questionable functions,
    so it should be as error-free as possible. Call it from an HTTP server
    if you get errors outside of the request.

    If extrabody is None, a friendly but rather unhelpful error message
    is set in the body. If extrabody is a string, it will be appended
    as-is to the body.
    """
    body = b'Unrecoverable error in the server.'
    if extrabody is not None:
        if not isinstance(extrabody, bytes):
            extrabody = extrabody.encode('utf-8')
        body += b'\n' + extrabody
    return (b'500 Internal Server Error', [(b'Content-Type', b'text/plain'), (b'Content-Length', ntob(str(len(body)), 'ISO-8859-1'))], [body])