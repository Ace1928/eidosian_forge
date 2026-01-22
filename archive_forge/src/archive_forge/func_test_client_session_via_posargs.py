from unittest import mock
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient.tests.unit import utils
from ironicclient.v1 import client
def test_client_session_via_posargs(self, http_client_mock):
    session = mock.Mock()
    client.Client('http://example.com', session)
    http_client_mock.assert_called_once_with(session, api_version_select_state='default', endpoint_override='http://example.com', os_ironic_api_version=client.DEFAULT_VER)