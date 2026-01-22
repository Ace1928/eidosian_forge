from unittest import mock
import ddt
import manilaclient
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient import exceptions
from manilaclient.tests.unit import utils
def test_requested_version_is_downgraded(self):
    server_end_version = '2.7'
    self._mock_returned_server_version(server_end_version, '2.0')
    max_version = api_versions.APIVersion('2.8')
    manilaclient.API_MAX_VERSION = max_version
    manilaclient.API_MIN_VERSION = api_versions.APIVersion('2.5')
    version = api_versions.discover_version(self.fake_client, max_version)
    self.assertEqual(api_versions.APIVersion(server_end_version), version)