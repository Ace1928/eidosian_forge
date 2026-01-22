import ddt
from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@ddt.data(True, False)
def test_clusters_list_is_up(self, detailed):
    lst = cs.clusters.list(is_up=True, detailed=detailed)
    self._assert_call('/clusters', detailed, 'is_up=True')
    self.assertEqual(2, len(lst))
    self._assert_request_id(lst)
    self._check_fields_present(lst, detailed)