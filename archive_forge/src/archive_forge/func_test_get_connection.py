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
def test_get_connection(self):
    endpoint = 'https://magnum-host:6385'
    client = http.HTTPClient(endpoint)
    conn = client.get_connection()
    self.assertTrue(conn, http.VerifiedHTTPSConnection)