from boto.compat import http_client
from tests.compat import mock, unittest
def assert_request_parameters(self, params, ignore_params_values=None):
    """Verify the actual parameters sent to the service API."""
    request_params = self.actual_request.params.copy()
    if ignore_params_values is not None:
        for param in ignore_params_values:
            try:
                del request_params[param]
            except KeyError:
                pass
    self.assertDictEqual(request_params, params)