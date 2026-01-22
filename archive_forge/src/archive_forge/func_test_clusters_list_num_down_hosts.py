import ddt
from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@ddt.data(True, False)
def test_clusters_list_num_down_hosts(self, detailed):
    lst = cs.clusters.list(num_down_hosts=2, detailed=detailed)
    self._assert_call('/clusters', detailed, 'num_down_hosts=2')
    self.assertEqual(2, len(lst))
    self._assert_request_id(lst)
    self._check_fields_present(lst, detailed)