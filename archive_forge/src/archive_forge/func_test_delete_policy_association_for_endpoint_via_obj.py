import uuid
from keystoneclient.tests.unit.v3 import test_endpoint_filter
from keystoneclient.tests.unit.v3 import utils
def test_delete_policy_association_for_endpoint_via_obj(self):
    self._crud_policy_association_for_endpoint_via_obj('DELETE', self.manager.delete_policy_association_for_endpoint)