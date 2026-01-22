import webob
from heat.api.middleware import version_negotiation as vn
from heat.tests import common
def test_request_path_contains_unknown_version(self):
    version_negotiation = vn.VersionNegotiationFilter(self._version_controller_factory, None, None)
    request = webob.Request({'PATH_INFO': 'v2.0/resource'})
    response = version_negotiation.process_request(request)
    self.assertIsInstance(response, VersionController)