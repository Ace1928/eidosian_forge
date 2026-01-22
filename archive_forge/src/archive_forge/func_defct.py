import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
@cherrypy.expose
def defct(self, newct):
    newct = 'text/%s' % newct
    cherrypy.config.update({'tools.response_headers.on': True, 'tools.response_headers.headers': [('Content-Type', newct)]})