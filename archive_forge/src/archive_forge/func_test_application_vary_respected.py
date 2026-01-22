from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_application_vary_respected(self):
    """Assert that an application's provided Vary header is persisted.

        If the underlying application, via middleware or other, provides a
        Vary header, its response should be honored.
        """
    request = webob.Request.blank('/server_cors_vary')
    request.method = 'GET'
    request.headers['Origin'] = 'http://valid.example.com'
    request.headers['Access-Control-Request-Method'] = 'GET'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin='http://valid.example.com', max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None, vary='Custom-Vary,Origin', has_content_type=True)