from unittest import mock
import novaclient
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient import utils as nutils
from novaclient.v2 import versions
def test_microversion_is_not_specified(self):
    response = mock.MagicMock(headers={api_versions.LEGACY_HEADER_NAME: ''})
    api_versions.check_headers(response, api_versions.APIVersion('2.2'))
    self.assertFalse(self.mock_log.warning.called)
    response = mock.MagicMock(headers={})
    api_versions.check_headers(response, api_versions.APIVersion('2.0'))
    self.assertFalse(self.mock_log.warning.called)