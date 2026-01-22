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
def test_server_redirect_exception(self):
    fake_redirect_resp = utils.FakeResponse({'content-type': 'application/octet-stream'}, 'foo', version=1, status=301)
    fake_resp = utils.FakeResponse({'content-type': 'application/octet-stream'}, 'bar', version=1, status=300)
    client = http.HTTPClient('http://localhost/')
    conn = utils.FakeConnection(fake_redirect_resp, redirect_resp=fake_resp)
    client.get_connection = lambda *a, **kw: conn
    self.assertRaises(MultipleChoices, client.json_request, 'GET', '/v1/resources')