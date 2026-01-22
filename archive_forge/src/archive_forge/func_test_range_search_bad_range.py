from openstack import exceptions
from openstack.tests.functional import base
def test_range_search_bad_range(self):
    flavors = self.user_cloud.list_flavors(get_extra=False)
    self.assertRaises(exceptions.SDKException, self.user_cloud.range_search, flavors, {'ram': '<1a0'})