from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
def handler_dec(f):

    @wraps(f)
    def wrapper(handler, *args, **kwargs):
        return f(handler, *args, **kwargs)
    return wrapper