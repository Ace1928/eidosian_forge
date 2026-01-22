import sys
from xmlrpc.client import (
import cherrypy
from cherrypy._cpcompat import ntob
def patched_path(path):
    """Return 'path', doctored for RPC."""
    if not path.endswith('/'):
        path += '/'
    if path.startswith('/RPC2/'):
        path = path[5:]
    return path