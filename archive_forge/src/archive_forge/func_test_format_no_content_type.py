import testtools
from unittest import mock
from glanceclient import exc
def test_format_no_content_type(self):
    mock_resp = mock.Mock()
    mock_resp.status_code = 400
    mock_resp.headers = {'content-type': 'application/octet-stream'}
    body = b'Error \n\n'
    err = exc.from_response(mock_resp, body)
    self.assertEqual('Error \n', err.details)