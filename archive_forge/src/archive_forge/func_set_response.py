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
def set_response(self):
    """Modify cherrypy.response status, headers, and body to represent
        self.

        CherryPy uses this internally, but you can also use it to create an
        HTTPError object and set its output without *raising* the exception.
        """
    response = cherrypy.serving.response
    clean_headers(self.code)
    response.status = self.status
    tb = None
    if cherrypy.serving.request.show_tracebacks:
        tb = format_exc()
    response.headers.pop('Content-Length', None)
    content = self.get_error_page(self.status, traceback=tb, message=self._message)
    response.body = content
    _be_ie_unfriendly(self.code)