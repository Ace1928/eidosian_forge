import random
import string
from openstack import exceptions
from openstack.tests.functional import base
def test_list_role_assignments_v2(self):
    user = self.operator_cloud.get_user('demo')
    project = self.operator_cloud.get_project('demo')
    assignments = self.operator_cloud.list_role_assignments(filters={'user': user['id'], 'project': project['id']})
    self.assertIsInstance(assignments, list)
    self.assertGreater(len(assignments), 0)