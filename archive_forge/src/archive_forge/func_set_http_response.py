from boto.compat import http_client
from tests.compat import mock, unittest
def set_http_response(self, status_code, reason='', header=[], body=None):
    http_response = self.create_response(status_code, reason, header, body)
    self.https_connection.getresponse.return_value = http_response