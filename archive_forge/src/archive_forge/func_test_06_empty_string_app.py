import sys
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.test import helper
def test_06_empty_string_app(self):
    if not cherrypy.server.using_wsgi:
        return self.skip('skipped (not using WSGI)... ')
    self.getPage('/hosted/app3')
    self.assertHeader('Content-Type', 'text/plain')
    self.assertInBody('Hello world')