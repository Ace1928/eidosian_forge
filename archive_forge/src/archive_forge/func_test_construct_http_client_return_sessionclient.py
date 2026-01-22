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
def test_construct_http_client_return_sessionclient(self):
    fake_session = mock.MagicMock()
    client = http._construct_http_client(session=fake_session)
    self.assertIsInstance(client, http.SessionClient)