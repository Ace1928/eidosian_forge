from openstack.object_store.v1 import obj
from openstack.tests.unit.cloud import test_object as base_test_object
def test_create_no_data(self):
    self._test_create('PUT', None)