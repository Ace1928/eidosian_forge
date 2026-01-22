from cinderclient.tests.unit.fixture_data import client
from cinderclient.tests.unit.fixture_data import snapshots
from cinderclient.tests.unit import utils
def test_snapshot_unmanage(self):
    s = self.cs.volume_snapshots.get('1234')
    self._assert_request_id(s)
    snap = self.cs.volume_snapshots.unmanage(s)
    self.assert_called('POST', '/snapshots/1234/action', {'os-unmanage': None})
    self._assert_request_id(snap)