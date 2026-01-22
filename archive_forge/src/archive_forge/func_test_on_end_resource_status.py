import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def test_on_end_resource_status(self):
    self.getPage('/status/on_end_resource_stage')
    self.assertBody('[]')
    self.getPage('/status/on_end_resource_stage')
    self.assertBody(repr(['200 OK']))