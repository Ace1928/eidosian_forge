import webob
from heat.api.middleware import version_negotiation as vn
from heat.tests import common
def test_match_version_string(self):
    version_negotiation = vn.VersionNegotiationFilter(self._version_controller_factory, None, None)
    request = webob.Request({})
    major_version = 1
    minor_version = 0
    match = version_negotiation._match_version_string('v{0}.{1}'.format(major_version, minor_version), request)
    self.assertTrue(match)
    self.assertEqual(major_version, request.environ['api.major_version'])
    self.assertEqual(minor_version, request.environ['api.minor_version'])