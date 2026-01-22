import webob
from heat.api.middleware import version_negotiation as vn
from heat.tests import common
def test_no_URI_version_accept_header_contains_invalid_MIME_type(self):
    version_negotiation = vn.VersionNegotiationFilter(self._version_controller_factory, None, None)
    request = webob.Request({'PATH_INFO': 'resource'})
    request.headers['Accept'] = 'application/invalidMIMEType'
    response = version_negotiation.process_request(request)
    self.assertIsInstance(response, webob.exc.HTTPNotFound)