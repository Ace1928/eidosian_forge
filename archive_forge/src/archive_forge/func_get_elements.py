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
def get_elements(self, headername):
    e = cherrypy.request.headers.elements(headername)
    return '\n'.join([str(x) for x in e])