from http import client as http_client
import io
from unittest import mock
from oslo_serialization import jsonutils
import socket
from magnumclient.common.apiclient.exceptions import GatewayTimeout
from magnumclient.common.apiclient.exceptions import MultipleChoices
from magnumclient.common import httpclient as http
from magnumclient import exceptions as exc
from magnumclient.tests import utils
def test_server_exception_empty_body(self):
    error_body = _get_error_body()
    fake_session = utils.FakeSession({'Content-Type': 'application/json'}, error_body, 500)
    client = http.SessionClient(session=fake_session)
    error = self.assertRaises(exc.InternalServerError, client.json_request, 'GET', '/v1/resources')
    self.assertEqual('Internal Server Error (HTTP 500)', str(error))