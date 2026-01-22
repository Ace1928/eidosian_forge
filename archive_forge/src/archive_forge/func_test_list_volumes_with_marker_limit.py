from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_list_volumes_with_marker_limit(self):
    lst = cs.volumes.list(marker=1234, limit=2)
    cs.assert_called('GET', '/volumes/detail?limit=2&marker=1234')
    self._assert_request_id(lst)