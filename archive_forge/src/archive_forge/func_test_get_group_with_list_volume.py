import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_get_group_with_list_volume(self):
    group_id = '1234'
    grp = cs.groups.get(group_id, list_volume=True)
    cs.assert_called('GET', '/groups/%s?list_volume=True' % group_id)
    self._assert_request_id(grp)