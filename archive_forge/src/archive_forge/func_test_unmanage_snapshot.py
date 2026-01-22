from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshots
def test_unmanage_snapshot(self):
    snapshot = 'fake_snapshot'
    version = api_versions.APIVersion('2.12')
    mock_microversion = mock.Mock(api_version=version)
    manager = share_snapshots.ShareSnapshotManager(api=mock_microversion)
    with mock.patch.object(manager, '_action', mock.Mock(return_value='fake')):
        result = manager.unmanage(snapshot)
        manager._action.assert_called_once_with('unmanage', snapshot)
        self.assertEqual('fake', result)