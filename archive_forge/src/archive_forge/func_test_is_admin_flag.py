import webob
from glance.api.middleware import context
import glance.context
from glance.tests.unit import base
def test_is_admin_flag(self):
    req = self._build_request(roles=['admin', 'role2'])
    self._build_middleware().process_request(req)
    self.assertTrue(req.context.is_admin)
    req = self._build_request()
    self._build_middleware().process_request(req)
    self.assertFalse(req.context.is_admin)
    from oslo_config.cfg import NoSuchOptError
    self.assertRaises(NoSuchOptError, self.config, admin_role='role1')