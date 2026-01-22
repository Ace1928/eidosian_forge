from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_wildcard_domain(self):
    """CORS Specification, Wildcards

        If the configuration file specifies CORS settings for the wildcard '*'
        domain, it should return those for all origin domains except for the
        overrides.
        """
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://default.example.com'
    request.headers['Access-Control-Request-Method'] = 'GET'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin='http://default.example.com', max_age=None, allow_methods='GET', allow_headers='', allow_credentials='true', expose_headers=None)
    request = webob.Request.blank('/')
    request.method = 'GET'
    request.headers['Origin'] = 'http://default.example.com'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin='http://default.example.com', max_age=None, allow_headers='', allow_credentials='true', expose_headers=None, has_content_type=True)
    request = webob.Request.blank('/')
    request.method = 'OPTIONS'
    request.headers['Origin'] = 'http://invalid.example.com'
    request.headers['Access-Control-Request-Method'] = 'GET'
    response = request.get_response(self.application)
    self.assertCORSResponse(response, status='200 OK', allow_origin='*', max_age=None, allow_methods='GET', allow_headers='', allow_credentials='true', expose_headers=None, has_content_type=True)