import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
def log_traceback(severity=logging.ERROR, debug=False):
    """Write the last error's traceback to the cherrypy error log."""
    cherrypy.log('', 'HTTP', severity=severity, traceback=True)