import uuid
from osc_placement.tests.functional import base
def test_list_no_resource_specified_error(self):
    self.assertCommandFailed('At least one --resource must be specified', self.openstack, 'allocation candidate list')