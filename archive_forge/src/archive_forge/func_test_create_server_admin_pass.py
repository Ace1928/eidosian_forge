import base64
import io
import os
import tempfile
from unittest import mock
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import floatingips
from novaclient.tests.unit.fixture_data import servers as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
def test_create_server_admin_pass(self):
    test_password = 'test-pass'
    test_key = 'fakekey'
    s = self.cs.servers.create(name='My server', image=1, flavor=1, admin_pass=test_password, key_name=test_key, nics=self._get_server_create_default_nics())
    self.assert_request_id(s, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('POST', '/servers')
    self.assertIsInstance(s, servers.Server)
    body = self.requests_mock.last_request.json()
    self.assertEqual(test_password, body['server']['adminPass'])