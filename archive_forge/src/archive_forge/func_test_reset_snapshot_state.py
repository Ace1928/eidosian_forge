from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshots
@ddt.data(('2.6', type('SnapshotUUID', (object,), {'uuid': '1234'})), ('2.7', type('SnapshotUUID', (object,), {'uuid': '1234'})), ('2.6', type('SnapshotID', (object,), {'id': '1234'})), ('2.7', type('SnapshotID', (object,), {'id': '1234'})), ('2.6', '1234'), ('2.7', '1234'))
@ddt.unpack
def test_reset_snapshot_state(self, microversion, snapshot):
    manager = self._get_manager(microversion)
    state = 'available'
    if api_versions.APIVersion(microversion) > api_versions.APIVersion('2.6'):
        action_name = 'reset_status'
    else:
        action_name = 'os-reset_status'
    with mock.patch.object(manager, '_action', mock.Mock()):
        manager.reset_state(snapshot, state)
        manager._action.assert_called_once_with(action_name, snapshot, {'status': state})