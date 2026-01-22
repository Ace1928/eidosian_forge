from wsgiref import util
from oslotest import base as test_base
import webob
from oslo_middleware import http_proxy_to_wsgi
def test_forwarded_for_headers(self):

    @webob.dec.wsgify()
    def fake_app(req):
        return req.environ['REMOTE_ADDR']
    self.middleware = http_proxy_to_wsgi.HTTPProxyToWSGI(fake_app)
    forwarded_for_addr = '1.2.3.4'
    forwarded_addr = '8.8.8.8'
    self.request.headers['Forwarded'] = 'for=%s;proto=https;host=example.com:8043' % forwarded_addr
    self.request.headers['X-Forwarded-For'] = forwarded_for_addr
    response = self.request.get_response(self.middleware)
    self.assertEqual(forwarded_addr.encode(), response.body)
    del self.request.headers['Forwarded']
    response = self.request.get_response(self.middleware)
    self.assertEqual(forwarded_for_addr.encode(), response.body)