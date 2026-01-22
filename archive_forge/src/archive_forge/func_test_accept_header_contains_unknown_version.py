import webob
from heat.api.middleware import version_negotiation as vn
from heat.tests import common
def test_accept_header_contains_unknown_version(self):
    version_negotiation = vn.VersionNegotiationFilter(self._version_controller_factory, None, None)
    request = webob.Request({'PATH_INFO': 'resource'})
    request.headers['Accept'] = 'application/vnd.openstack.orchestration-v2.0'
    response = version_negotiation.process_request(request)
    self.assertIsInstance(response, VersionController)