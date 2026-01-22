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
@mock.patch.object(http.VersionNegotiationMixin, '_make_simple_request', autospec=True)
@mock.patch.object(http.VersionNegotiationMixin, '_parse_version_headers', autospec=True)
def test_negotiate_version_server_no_version_on_error(self, mock_pvh, mock_msr, mock_save_data):
    mock_pvh.side_effect = iter([(None, None), ('1.1', '1.2')])
    mock_conn = mock.MagicMock()
    result = self.test_object.negotiate_version(mock_conn, self.response)
    self.assertEqual('1.2', result)
    self.assertTrue(mock_msr.called)
    self.assertEqual(2, mock_pvh.call_count)
    self.assertEqual(1, mock_save_data.call_count)