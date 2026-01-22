import webob
from glance.api.middleware import context
import glance.context
from glance.tests.unit import base
def test_anonymous_access_defaults_to_disabled(self):
    req = self._build_request(identity_status='Nope')
    middleware = self._build_middleware()
    self.assertRaises(webob.exc.HTTPUnauthorized, middleware.process_request, req)