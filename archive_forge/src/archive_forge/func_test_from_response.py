from http import client as http_client
from unittest import mock
from ironicclient.common.apiclient import exceptions
from ironicclient import exc
from ironicclient.tests.unit import utils as test_utils
def test_from_response(self, mock_apiclient):
    fake_response = mock.Mock(status_code=http_client.BAD_REQUEST)
    exc.from_response(fake_response, message=self.message, traceback=self.traceback, method=self.method, url=self.url)
    self.assertEqual(http_client.BAD_REQUEST, fake_response.status_code)
    self.assertEqual(self.expected_json, fake_response.json())
    mock_apiclient.assert_called_once_with(fake_response, method=self.method, url=self.url)