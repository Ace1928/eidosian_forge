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
def test_server_body_undecode_json(self):
    err = 'foo'
    fake_resp = utils.FakeResponse({'content-type': 'application/json'}, io.StringIO(err), version=1, status=200)
    client = http.HTTPClient('http://localhost/')
    conn = utils.FakeConnection(fake_resp)
    client.get_connection = lambda *a, **kw: conn
    resp, body = client.json_request('GET', '/v1/resources')
    self.assertEqual(resp, fake_resp)
    self.assertEqual(err, body)