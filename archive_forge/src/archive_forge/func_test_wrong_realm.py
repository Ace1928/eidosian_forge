import cherrypy
from cherrypy.lib import auth_digest
from cherrypy._cpcompat import ntob
from cherrypy.test import helper
def test_wrong_realm(self):
    self._test_parametric_digest(username='test', realm='wrong realm')
    assert self.status_code == 401