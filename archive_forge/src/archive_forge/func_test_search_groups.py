from openstack import exceptions
from openstack.tests.functional import base
def test_search_groups(self):
    group_name = self.group_prefix + '_search'
    results = self.operator_cloud.search_groups(filters=dict(name=group_name))
    self.assertEqual(0, len(results))
    group = self.operator_cloud.create_group(group_name, 'test group')
    self.assertEqual(group_name, group['name'])
    results = self.operator_cloud.search_groups(filters=dict(name=group_name))
    self.assertEqual(1, len(results))
    self.assertEqual(group_name, results[0]['name'])