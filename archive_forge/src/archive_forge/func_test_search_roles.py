import random
import string
from openstack import exceptions
from openstack.tests.functional import base
def test_search_roles(self):
    roles = self.operator_cloud.search_roles(filters={'name': 'admin'})
    self.assertIsNotNone(roles)
    self.assertEqual(1, len(roles))
    self.assertEqual('admin', roles[0]['name'])