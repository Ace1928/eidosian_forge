import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
def log_request_headers(debug=False):
    """Write request headers to the cherrypy error log."""
    h = ['  %s: %s' % (k, v) for k, v in cherrypy.serving.request.header_list]
    cherrypy.log('\nRequest Headers:\n' + '\n'.join(h), 'HTTP')