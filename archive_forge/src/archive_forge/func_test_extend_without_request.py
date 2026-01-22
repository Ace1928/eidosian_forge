import webob
from oslo_middleware.base import ConfigurableMiddleware
from oslo_middleware.base import Middleware
from oslotest.base import BaseTestCase
def test_extend_without_request(self):
    """Assert that an older middleware behaves as appropriate.

        This tests makes sure that the request method is NOT passed to the
        middleware's implementation, and that there are no other expected
        errors.
        """
    self.application = NoRequestBase(application)
    request = webob.Request({}, method='GET')
    request.get_response(self.application)
    self.assertTrue(self.application.called_without_request)