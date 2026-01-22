import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
def referer(pattern, accept=True, accept_missing=False, error=403, message='Forbidden Referer header.', debug=False):
    """Raise HTTPError if Referer header does/does not match the given pattern.

    pattern
        A regular expression pattern to test against the Referer.

    accept
        If True, the Referer must match the pattern; if False,
        the Referer must NOT match the pattern.

    accept_missing
        If True, permit requests with no Referer header.

    error
        The HTTP error code to return to the client on failure.

    message
        A string to include in the response body on failure.

    """
    try:
        ref = cherrypy.serving.request.headers['Referer']
        match = bool(re.match(pattern, ref))
        if debug:
            cherrypy.log('Referer %r matches %r' % (ref, pattern), 'TOOLS.REFERER')
        if accept == match:
            return
    except KeyError:
        if debug:
            cherrypy.log('No Referer header', 'TOOLS.REFERER')
        if accept_missing:
            return
    raise cherrypy.HTTPError(error, message)