from cinderclient.tests.unit.fixture_data import client
from cinderclient.tests.unit.fixture_data import snapshots
from cinderclient.tests.unit import utils
def test_update_snapshot_status(self):
    snap = self.cs.volume_snapshots.get('1234')
    self._assert_request_id(snap)
    stat = {'status': 'available'}
    stats = self.cs.volume_snapshots.update_snapshot_status(snap, stat)
    self.assert_called('POST', '/snapshots/1234/action')
    self._assert_request_id(stats)