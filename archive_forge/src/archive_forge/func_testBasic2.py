from hashlib import md5
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.lib import auth_basic
from cherrypy.test import helper
def testBasic2(self):
    self.getPage('/basic2/')
    self.assertStatus(401)
    self.assertHeader('WWW-Authenticate', 'Basic realm="wonderland"')
    self.getPage('/basic2/', [('Authorization', 'Basic eHVzZXI6eHBhc3N3b3JX')])
    self.assertStatus(401)
    self.getPage('/basic2/', [('Authorization', 'Basic eHVzZXI6eHBhc3N3b3Jk')])
    self.assertStatus('200 OK')
    self.assertBody("Hello xuser, you've been authorized.")