import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
@cherrypy.config(**{'tools.log_status.on': True})
def on_end_resource_stage(self):
    return repr(self.statuses)