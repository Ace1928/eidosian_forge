import json
import logging
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as keystone_exception
from oslo_serialization import jsonutils
from cinderclient import api_versions
import cinderclient.client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@mock.patch('cinderclient.client.requests.get')
def test_get_server_version_cacert(self, mock_request):
    mock_response = utils.TestResponse({'status_code': 200, 'text': json.dumps(fakes.fake_request_get_no_v3())})
    mock_request.return_value = mock_response
    url = 'https://192.168.122.127:8776/v3/e5526285ebd741b1819393f772f11fc3'
    expected_url = 'https://192.168.122.127:8776/'
    cacert = '/path/to/cert'
    cinderclient.client.get_server_version(url, cacert=cacert)
    mock_request.assert_called_once_with(expected_url, verify=cacert, cert=None)