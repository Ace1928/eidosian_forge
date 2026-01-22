from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_list_consistencygroup_with_empty_search_opt(self):
    lst = cs.consistencygroups.list(search_opts={'foo': 'bar', 'abc': None})
    cs.assert_called('GET', '/consistencygroups/detail?foo=bar')
    self._assert_request_id(lst)