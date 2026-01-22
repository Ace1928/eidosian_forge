from unittest import mock
import ddt
import manilaclient
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient import exceptions
from manilaclient.tests.unit import utils
def test_requested_version_in_range(self):
    self._mock_returned_server_version('2.7', '2.4')
    manilaclient.API_MAX_VERSION = api_versions.APIVersion('2.11')
    manilaclient.API_MIN_VERSION = api_versions.APIVersion('2.1')
    discovered_version = api_versions.discover_version(self.fake_client, api_versions.APIVersion('2.7'))
    self.assertEqual('2.7', discovered_version.get_string())
    self.assertTrue(self.fake_client.services.server_api_version.called)