import webob
from heat.api.middleware import version_negotiation as vn
from heat.tests import common
def test_invalid_utf8_path(self):
    version_negotiation = vn.VersionNegotiationFilter(self._version_controller_factory, None, None)
    request = webob.Request.blank('/%c0')
    response = version_negotiation.process_request(request)
    self.assertIsInstance(response, webob.exc.HTTPBadRequest)