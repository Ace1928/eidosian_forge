from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_no_origin_fail(self):
    """Assert that a filter factory with no allowed_origin fails."""
    self.assertRaises(TypeError, cors.filter_factory, global_conf=None, allow_credentials='False', max_age='', expose_headers='', allow_methods='GET', allow_headers='')