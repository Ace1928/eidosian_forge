import webob
from heat.api.middleware import version_negotiation as vn
from heat.tests import common
def test_removes_version_from_request_path(self):
    version_negotiation = vn.VersionNegotiationFilter(self._version_controller_factory, None, None)
    expected_path = 'resource'
    request = webob.Request({'PATH_INFO': 'v1.0/{0}'.format(expected_path)})
    response = version_negotiation.process_request(request)
    self.assertIsNone(response)
    self.assertEqual(expected_path, request.path_info_peek())