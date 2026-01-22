from wsgiref import util
from oslotest import base as test_base
import webob
from oslo_middleware import http_proxy_to_wsgi
def test_url_translate_port_and_base_and_proto_and_host(self):
    self.request.headers['X-Forwarded-Proto'] = 'https'
    self.request.headers['X-Forwarded-Prefix'] = '/bla'
    self.request.headers['X-Forwarded-Host'] = 'example.com:8043'
    response = self.request.get_response(self.middleware)
    self.assertEqual(b'https://example.com:8043/bla', response.body)