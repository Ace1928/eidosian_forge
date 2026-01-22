from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_optional_max_age(self):
    """CORS Specification Section 6.2.8

        Optionally add a single Access-Control-Max-Age header with as value
        the amount of seconds the user agent is allowed to cache the result of
        the request.
        """
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://cached.example.com'
    request.headers['Access-Control-Request-Method'] = 'GET'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin='http://cached.example.com', max_age=3600, allow_methods='GET', allow_headers=None, allow_credentials=None, expose_headers=None)