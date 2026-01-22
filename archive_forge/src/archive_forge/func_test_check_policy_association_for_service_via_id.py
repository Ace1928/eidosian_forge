import uuid
from keystoneclient.tests.unit.v3 import test_endpoint_filter
from keystoneclient.tests.unit.v3 import utils
def test_check_policy_association_for_service_via_id(self):
    self._crud_policy_association_for_service_via_id('HEAD', self.manager.check_policy_association_for_service)