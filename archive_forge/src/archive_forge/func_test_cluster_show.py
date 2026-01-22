import ddt
from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_cluster_show(self):
    result = cs.clusters.show('1')
    self._assert_call('/clusters/1', False)
    self._assert_request_id(result)
    self._check_fields_present([result], True)