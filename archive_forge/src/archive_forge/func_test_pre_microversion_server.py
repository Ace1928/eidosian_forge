from unittest import mock
import ddt
import manilaclient
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient import exceptions
from manilaclient.tests.unit import utils
def test_pre_microversion_server(self):
    self.fake_client.services.server_api_version.return_value = []
    manilaclient.API_MAX_VERSION = api_versions.APIVersion('2.5')
    manilaclient.API_MIN_VERSION = api_versions.APIVersion('2.5')
    discovered_version = api_versions.discover_version(self.fake_client, manilaclient.API_MAX_VERSION)
    self.assertEqual('1.0', discovered_version.get_string())
    self.assertTrue(self.fake_client.services.server_api_version.called)