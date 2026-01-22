from openstack import exceptions
from openstack.tests.functional import base
def test_range_search_lt(self):
    flavors = self.user_cloud.list_flavors(get_extra=False)
    result = self.user_cloud.range_search(flavors, {'ram': '<1024'})
    self.assertIsInstance(result, list)
    result = self._filter_m1_flavors(result)
    self.assertEqual(1, len(result))
    self.assertEqual('m1.tiny', result[0]['name'])