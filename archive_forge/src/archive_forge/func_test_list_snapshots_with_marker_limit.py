from cinderclient.tests.unit.fixture_data import client
from cinderclient.tests.unit.fixture_data import snapshots
from cinderclient.tests.unit import utils
def test_list_snapshots_with_marker_limit(self):
    lst = self.cs.volume_snapshots.list(marker=1234, limit=2)
    self.assert_called('GET', '/snapshots/detail?limit=2&marker=1234')
    self._assert_request_id(lst)