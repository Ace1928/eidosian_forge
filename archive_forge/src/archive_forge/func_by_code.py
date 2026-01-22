import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
@cherrypy.config(**{'tools.trailing_slash.extra': True})
def by_code(self, code):
    raise cherrypy.HTTPRedirect('somewhere%20else', code)