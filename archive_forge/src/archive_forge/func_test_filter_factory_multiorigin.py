from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_filter_factory_multiorigin(self):
    self.config([])
    filter = cors.filter_factory(None, allowed_origin='http://valid.example.com,http://other.example.com')
    application = filter(test_application)
    self.assertIn('http://valid.example.com', application.allowed_origins)
    self.assertIn('http://other.example.com', application.allowed_origins)