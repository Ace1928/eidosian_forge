from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def test_client_end_version_is_the_latest_one(self):
    fake_client = mock.MagicMock()
    fake_client.versions.get_current.return_value = mock.MagicMock(max_version='1.16', min_version='1.4')
    api_versions.MAX_API_VERSION = '1.11'
    api_versions.MIN_API_VERSION = '1.1'
    self.assertEqual('1.11', api_versions.discover_version(fake_client, api_versions.APIVersion('1.latest')).get_string())