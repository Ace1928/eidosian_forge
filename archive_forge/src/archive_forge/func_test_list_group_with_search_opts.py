import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_list_group_with_search_opts(self):
    lst = cs.groups.list(search_opts={'foo': 'bar'})
    cs.assert_called('GET', '/groups/detail?foo=bar')
    self._assert_request_id(lst)