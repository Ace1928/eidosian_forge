import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def slice_file(self):
    path = os.path.join(os.getcwd(), os.path.dirname(__file__))
    return static.serve_file(os.path.join(path, 'static/index.html'))