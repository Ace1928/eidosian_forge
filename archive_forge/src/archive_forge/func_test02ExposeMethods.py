import sys
import importlib
import cherrypy
from cherrypy.test import helper
def test02ExposeMethods(self):
    self.setup_tutorial('tut02_expose_methods', 'HelloWorld')
    self.getPage('/show_msg')
    self.assertBody('Hello world!')