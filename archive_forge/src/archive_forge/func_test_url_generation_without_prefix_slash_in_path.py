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
def test_url_generation_without_prefix_slash_in_path(self):
    client = http.HTTPClient('http://localhost')
    url = client._make_connection_url('v1/resources')
    self.assertEqual('/v1/resources', url)