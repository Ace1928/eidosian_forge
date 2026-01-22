import cherrypy
from cherrypy.lib import auth_digest
from cherrypy._cpcompat import ntob
from cherrypy.test import helper
def testPublic(self):
    self.getPage('/')
    assert self.status == '200 OK'
    self.assertHeader('Content-Type', 'text/html;charset=utf-8')
    assert self.body == b'This is public.'