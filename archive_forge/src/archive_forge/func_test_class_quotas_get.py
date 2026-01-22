from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_class_quotas_get(self):
    class_name = 'test'
    cls = cs.quota_classes.get(class_name)
    cs.assert_called('GET', '/os-quota-class-sets/%s' % class_name)
    self._assert_request_id(cls)