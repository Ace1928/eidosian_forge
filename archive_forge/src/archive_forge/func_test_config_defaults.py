from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_config_defaults(self):
    """Assert that using set_defaults overrides the appropriate values."""
    cors.set_defaults(**self.override_opts)
    for opt in cors.CORS_OPTS:
        if opt.dest in self.override_opts:
            self.assertEqual(self.override_opts[opt.dest], opt.default)