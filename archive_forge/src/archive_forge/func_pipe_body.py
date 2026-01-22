import gzip
import io
import sys
import time
import types
import unittest
import operator
from http.client import IncompleteRead
import cherrypy
from cherrypy import tools
from cherrypy._cpcompat import ntou
from cherrypy.test import helper, _test_decorators
def pipe_body():
    cherrypy.request.process_request_body = False
    clen = int(cherrypy.request.headers['Content-Length'])
    cherrypy.request.body = cherrypy.request.rfile.read(clen)