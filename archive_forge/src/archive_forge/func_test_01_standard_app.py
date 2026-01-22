import sys
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.test import helper
def test_01_standard_app(self):
    self.getPage('/')
    self.assertBody("I'm a regular CherryPy page handler!")