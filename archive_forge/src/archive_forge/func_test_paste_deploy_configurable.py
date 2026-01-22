import webob
from oslo_middleware.base import ConfigurableMiddleware
from oslo_middleware.base import Middleware
from oslotest.base import BaseTestCase
def test_paste_deploy_configurable(self):
    app = ConfigurableMiddlewareTest.factory({'global': True}, local=True)(application)
    self.assertEqual({'global': True, 'local': True}, app.conf)