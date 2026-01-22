import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_list_group_detailed_false(self):
    lst = cs.groups.list(detailed=False)
    cs.assert_called('GET', '/groups')
    self._assert_request_id(lst)