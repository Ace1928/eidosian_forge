from wsgiref import util
from oslotest import base as test_base
import webob
from oslo_middleware import http_proxy_to_wsgi
def test_url_translate_ssl_port(self):
    self.request.headers['X-Forwarded-Proto'] = 'https'
    self.request.headers['X-Forwarded-Host'] = 'example.com:123'
    response = self.request.get_response(self.middleware)
    self.assertEqual(b'https://example.com:123/', response.body)