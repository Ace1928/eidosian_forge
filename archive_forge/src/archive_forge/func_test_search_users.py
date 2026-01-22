from openstack import exceptions
from openstack.tests.functional import base
def test_search_users(self):
    users = self.operator_cloud.search_users(filters={'is_enabled': True})
    self.assertIsNotNone(users)