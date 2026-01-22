from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_inherited_role_grants_for_user(self):
    """Test inherited user roles.

        Test Plan:

        - Enable OS-INHERIT extension
        - Create 3 roles
        - Create a domain, with a project and a user
        - Check no roles yet exit
        - Assign a direct user role to the project and a (non-inherited)
          user role to the domain
        - Get a list of effective roles - should only get the one direct role
        - Now add an inherited user role to the domain
        - Get a list of effective roles - should have two roles, one
          direct and one by virtue of the inherited user role
        - Also get effective roles for the domain - the role marked as
          inherited should not show up

        """
    role_list = []
    for _ in range(3):
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        role_list.append(role)
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    user1 = unit.new_user_ref(domain_id=domain1['id'])
    user1 = PROVIDERS.identity_api.create_user(user1)
    project1 = unit.new_project_ref(domain_id=domain1['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], project_id=project1['id'])
    self.assertEqual(0, len(roles_ref))
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project1['id'], role_id=role_list[0]['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=role_list[1]['id'])
    combined_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(user1['id'], project1['id'])
    self.assertEqual(1, len(combined_list))
    self.assertIn(role_list[0]['id'], combined_list)
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=role_list[2]['id'], inherited_to_projects=True)
    combined_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(user1['id'], project1['id'])
    self.assertEqual(2, len(combined_list))
    self.assertIn(role_list[0]['id'], combined_list)
    self.assertIn(role_list[2]['id'], combined_list)
    combined_role_list = PROVIDERS.assignment_api.get_roles_for_user_and_domain(user1['id'], domain1['id'])
    self.assertEqual(1, len(combined_role_list))
    self.assertIn(role_list[1]['id'], combined_role_list)
    test_plan = {'entities': {'domains': {'users': 1, 'projects': 1}, 'roles': 3}, 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 1, 'domain': 0}, {'user': 0, 'role': 2, 'domain': 0, 'inherited_to_projects': True}], 'tests': [{'params': {'user': 0, 'project': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 2, 'project': 0, 'indirect': {'domain': 0}}]}, {'params': {'user': 0, 'domain': 0, 'effective': True}, 'results': [{'user': 0, 'role': 1, 'domain': 0}]}, {'params': {'user': 0, 'domain': 0, 'inherited': False}, 'results': [{'user': 0, 'role': 1, 'domain': 0}]}]}
    self.execute_assignment_plan(test_plan)