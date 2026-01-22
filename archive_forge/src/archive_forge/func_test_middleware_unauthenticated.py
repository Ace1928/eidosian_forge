import base64
import os
import tempfile
from oslo_config import cfg
import webob
from oslo_middleware import basic_auth as auth
from oslotest import base as test_base
def test_middleware_unauthenticated(self):
    auth_file = self.write_auth_file('myName:$2y$05$lE3eGtyj41jZwrzS87KTqe6.JETVCWBkc32C63UP2aYrGoYOEpbJm\n\n\n')
    cfg.CONF.set_override('http_basic_auth_user_file', auth_file, group='oslo_middleware')
    self.middleware = auth.BasicAuthMiddleware(self.fake_app)
    response = self.request.get_response(self.middleware)
    self.assertEqual('401 Unauthorized', response.status)