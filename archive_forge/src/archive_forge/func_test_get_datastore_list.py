import io
import re
import sys
from unittest import mock
import fixtures
from keystoneauth1 import fixture
import requests_mock
import testtools
import uuid
import troveclient.client
from troveclient import exceptions
import troveclient.shell
@mock.patch('keystoneauth1.discover.get_version_data', return_value=[{'status': 'stable', 'id': version_id, 'links': links}])
@mock.patch('troveclient.v1.datastores.Datastores.list')
@requests_mock.Mocker()
def test_get_datastore_list(self, mock_discover, mock_list, mock_requests):
    expected = '\n'.join(['+----+------+', '| ID | Name |', '+----+------+', '+----+------+', ''])
    self.make_env()
    self.register_keystone_discovery_fixture(mock_requests)
    mock_requests.register_uri('POST', 'http://no.where/v3/auth/tokens', headers={'X-Subject-Token': 'fakeToken'}, text=self.v3_auth_response)
    stdout, stderr = self.shell('datastore-list')
    self.assertEqual(expected, stdout + stderr)