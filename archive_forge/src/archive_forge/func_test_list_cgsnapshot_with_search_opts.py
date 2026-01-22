from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_list_cgsnapshot_with_search_opts(self):
    lst = cs.cgsnapshots.list(search_opts={'foo': 'bar'})
    cs.assert_called('GET', '/cgsnapshots/detail?foo=bar')
    self._assert_request_id(lst)