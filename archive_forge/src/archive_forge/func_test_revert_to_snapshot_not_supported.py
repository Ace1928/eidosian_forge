from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
def test_revert_to_snapshot_not_supported(self):
    share = 'fake_share'
    snapshot = 'fake_snapshot'
    version = api_versions.APIVersion('2.26')
    mock_microversion = mock.Mock(api_version=version)
    manager = shares.ShareManager(api=mock_microversion)
    self.assertRaises(client_exceptions.UnsupportedVersion, manager.revert_to_snapshot, share, snapshot)