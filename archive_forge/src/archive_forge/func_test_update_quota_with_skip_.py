from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_update_quota_with_skip_(self):
    q = cs.quotas.get('test')
    q.update(skip_validation=False)
    cs.assert_called('PUT', '/os-quota-sets/test?skip_validation=False')
    self._assert_request_id(q)