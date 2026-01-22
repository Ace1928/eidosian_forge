import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
def log_hooks(debug=False):
    """Write request.hooks to the cherrypy error log."""
    request = cherrypy.serving.request
    msg = []
    from cherrypy import _cprequest
    points = _cprequest.hookpoints
    for k in request.hooks.keys():
        if k not in points:
            points.append(k)
    for k in points:
        msg.append('    %s:' % k)
        v = request.hooks.get(k, [])
        v.sort()
        for h in v:
            msg.append('        %r' % h)
    cherrypy.log('\nRequest Hooks for ' + cherrypy.url() + ':\n' + '\n'.join(msg), 'HTTP')