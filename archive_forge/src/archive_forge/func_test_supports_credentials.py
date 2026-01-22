from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_supports_credentials(self):
    """CORS Specification Section 6.1.3

        If the resource supports credentials add a single
        Access-Control-Allow-Origin header, with the value of the Origin header
        as value, and add a single Access-Control-Allow-Credentials header with
        the case-sensitive string "true" as value.

        Otherwise, add a single Access-Control-Allow-Origin header, with
        either the value of the Origin header or the string "*" as value.

        NOTE: We never use the "*" as origin.
        """
    for method in self.methods:
        request = webob.Request.blank('/')
        request.method = method
        request.headers['Origin'] = 'http://valid.example.com'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://valid.example.com', max_age=None, allow_methods=None, allow_headers=None, allow_credentials=None, expose_headers=None, has_content_type=True)
    for method in self.methods:
        request = webob.Request.blank('/')
        request.method = method
        request.headers['Origin'] = 'http://creds.example.com'
        response = request.get_response(self.application)
        self.assertCORSResponse(response, status='200 OK', allow_origin='http://creds.example.com', max_age=None, allow_methods=None, allow_headers=None, allow_credentials='true', expose_headers=None, has_content_type=True)