from unittest import mock
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient.tests.unit import utils
from ironicclient.v1 import client
def test_client_cache_version_no_endpoint_as_arg(self, http_client_mock):
    client.Client(session='fake_session', insecure=True)
    http_client_mock.assert_called_once_with(session='fake_session', insecure=True, os_ironic_api_version=client.DEFAULT_VER, api_version_select_state='default')