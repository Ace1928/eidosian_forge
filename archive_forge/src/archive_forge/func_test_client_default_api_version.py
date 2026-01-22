from unittest import mock
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient.tests.unit import utils
from ironicclient.v1 import client
@mock.patch.object(filecache, 'retrieve_data', autospec=True)
def test_client_default_api_version(self, cache_mock, http_client_mock):
    endpoint = 'http://ironic:6385'
    cache_mock.return_value = None
    client.Client(endpoint, session=self.session)
    cache_mock.assert_called_once_with(host='ironic', port='6385')
    http_client_mock.assert_called_once_with(endpoint_override=endpoint, session=self.session, os_ironic_api_version=client.DEFAULT_VER, api_version_select_state='default')