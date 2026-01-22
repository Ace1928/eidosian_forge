import webob
from oslo_middleware.base import ConfigurableMiddleware
from oslo_middleware.base import Middleware
from oslotest.base import BaseTestCase
def test_no_content_type_added(self):

    class TestMiddleware(Middleware):

        @staticmethod
        def process_request(req):
            return 'foobar'
    m = TestMiddleware(None)
    request = webob.Request({}, method='GET')
    response = request.get_response(m)
    self.assertNotIn('Content-Type', response.headers)