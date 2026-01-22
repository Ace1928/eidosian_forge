from unittest import mock
import ddt
from oslo_utils import uuidutils
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.v2 import client
@mock.patch.object(client.Client, '_get_keystone_client', mock.Mock())
def test_regions_with_same_name(self):
    self.mock_object(client.httpclient, 'HTTPClient')
    catalog = {'sharev2': [{'region': 'FirstRegion', 'publicURL': 'http://1.2.3.4'}, {'region': 'secondregion', 'publicURL': 'http://1.1.1.1'}, {'region': 'SecondRegion', 'publicURL': 'http://2.2.2.2'}]}
    kc = client.Client._get_keystone_client.return_value
    kc.service_catalog = mock.Mock()
    kc.service_catalog.get_endpoints = mock.Mock(return_value=catalog)
    c = client.Client(api_version=manilaclient.API_MIN_VERSION, service_type='sharev2', region_name='SecondRegion')
    self.assertTrue(client.Client._get_keystone_client.called)
    kc.service_catalog.get_endpoints.assert_called_with('sharev2')
    client.httpclient.HTTPClient.assert_called_with('http://2.2.2.2', mock.ANY, 'python-manilaclient', insecure=False, cacert=None, cert=None, timeout=None, retries=None, http_log_debug=False, api_version=manilaclient.API_MIN_VERSION)
    self.assertIsNotNone(c.client)