from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_multi_group_grants_on_project_domain(self):
    """Test multiple group roles for user on project and domain.

        Test Plan:

        - Create 6 roles
        - Create a domain, with a project, user and two groups
        - Make the user a member of both groups
        - Check no roles yet exit
        - Assign a role to each user and both groups on both the
          project and domain
        - Get a list of effective roles for the user on both the
          project and domain, checking we get back the correct three
          roles

        """
    role_list = []
    for _ in range(6):
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        role_list.append(role)
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    user1 = unit.new_user_ref(domain_id=domain1['id'])
    user1 = PROVIDERS.identity_api.create_user(user1)
    group1 = unit.new_group_ref(domain_id=domain1['id'])
    group1 = PROVIDERS.identity_api.create_group(group1)
    group2 = unit.new_group_ref(domain_id=domain1['id'])
    group2 = PROVIDERS.identity_api.create_group(group2)
    project1 = unit.new_project_ref(domain_id=domain1['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
    PROVIDERS.identity_api.add_user_to_group(user1['id'], group2['id'])
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], project_id=project1['id'])
    self.assertEqual(0, len(roles_ref))
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=role_list[0]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain1['id'], role_id=role_list[1]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group2['id'], domain_id=domain1['id'], role_id=role_list[2]['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project1['id'], role_id=role_list[3]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=project1['id'], role_id=role_list[4]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group2['id'], project_id=project1['id'], role_id=role_list[5]['id'])
    combined_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(user1['id'], project1['id'])
    self.assertEqual(3, len(combined_list))
    self.assertIn(role_list[3]['id'], combined_list)
    self.assertIn(role_list[4]['id'], combined_list)
    self.assertIn(role_list[5]['id'], combined_list)
    combined_role_list = PROVIDERS.assignment_api.get_roles_for_user_and_domain(user1['id'], domain1['id'])
    self.assertEqual(3, len(combined_role_list))
    self.assertIn(role_list[0]['id'], combined_role_list)
    self.assertIn(role_list[1]['id'], combined_role_list)
    self.assertIn(role_list[2]['id'], combined_role_list)