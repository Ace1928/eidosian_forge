from openstack import exceptions
from openstack.tests.functional import base
def test_range_search_multi_3(self):
    flavors = self.user_cloud.list_flavors(get_extra=False)
    result = self.user_cloud.range_search(flavors, {'ram': '>=4096', 'vcpus': '<6'})
    self.assertIsInstance(result, list)
    result = self._filter_m1_flavors(result)
    self.assertEqual(2, len(result))
    flavor_names = [r['name'] for r in result]
    self.assertIn('m1.medium', flavor_names)
    self.assertIn('m1.large', flavor_names)