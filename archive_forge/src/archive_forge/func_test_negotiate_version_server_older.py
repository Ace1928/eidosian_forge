from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
@mock.patch.object(filecache, 'save_data', autospec=True)
@mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
def test_negotiate_version_server_older(self, mock_pvh, mock_save_data):
    latest_ver = '1.5'
    mock_pvh.return_value = ('1.1', latest_ver)
    mock_conn = mock.MagicMock()
    result = self.test_object.negotiate_version(mock_conn, self.response)
    self.assertEqual(latest_ver, result)
    self.assertEqual(1, mock_pvh.call_count)
    host, port = http.get_server(self.test_object.endpoint_override)
    mock_save_data.assert_called_once_with(host=host, port=port, data=latest_ver)