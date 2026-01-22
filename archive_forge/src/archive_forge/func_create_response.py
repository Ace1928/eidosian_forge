from boto.compat import http_client
from tests.compat import mock, unittest
def create_response(self, status_code, reason='', header=[], body=None):
    if body is None:
        body = self.default_body()
    response = mock.Mock(spec=http_client.HTTPResponse)
    response.status = status_code
    response.read.return_value = body
    response.reason = reason
    response.getheaders.return_value = header
    response.msg = dict(header)

    def overwrite_header(arg, default=None):
        header_dict = dict(header)
        if arg in header_dict:
            return header_dict[arg]
        else:
            return default
    response.getheader.side_effect = overwrite_header
    return response