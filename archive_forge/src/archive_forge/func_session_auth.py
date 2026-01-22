import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
def session_auth(**kwargs):
    'Session authentication hook.\n\n    Any attribute of the SessionAuth class may be overridden\n    via a keyword arg to this function:\n\n    ' + '\n    '.join(('{!s}: {!s}'.format(k, type(getattr(SessionAuth, k)).__name__) for k in dir(SessionAuth) if not k.startswith('__')))
    sa = SessionAuth()
    for k, v in kwargs.items():
        setattr(sa, k, v)
    return sa.run()