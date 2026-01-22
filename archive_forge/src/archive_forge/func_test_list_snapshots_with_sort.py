from cinderclient.tests.unit.fixture_data import client
from cinderclient.tests.unit.fixture_data import snapshots
from cinderclient.tests.unit import utils
def test_list_snapshots_with_sort(self):
    lst = self.cs.volume_snapshots.list(sort='id')
    self.assert_called('GET', '/snapshots/detail?sort=id')
    self._assert_request_id(lst)