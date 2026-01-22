import webob
from glance.api.middleware import context
import glance.context
from glance.tests.unit import base
def test_header_parsing(self):
    req = self._build_request()
    self._build_middleware().process_request(req)
    self.assertEqual('token1', req.context.auth_token)
    self.assertEqual('user1', req.context.user_id)
    self.assertEqual('tenant1', req.context.project_id)
    self.assertEqual(['role1', 'role2'], req.context.roles)