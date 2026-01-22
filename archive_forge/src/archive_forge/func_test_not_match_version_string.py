import webob
from heat.api.middleware import version_negotiation as vn
from heat.tests import common
def test_not_match_version_string(self):
    version_negotiation = vn.VersionNegotiationFilter(self._version_controller_factory, None, None)
    request = webob.Request({})
    match = version_negotiation._match_version_string('invalid', request)
    self.assertFalse(match)