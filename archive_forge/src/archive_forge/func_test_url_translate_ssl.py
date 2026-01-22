from wsgiref import util
from oslotest import base as test_base
import webob
from oslo_middleware import http_proxy_to_wsgi
def test_url_translate_ssl(self):
    self.request.headers['X-Forwarded-Proto'] = 'https'
    response = self.request.get_response(self.middleware)
    self.assertEqual(b'https://localhost:80/', response.body)