from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_and_remove_correct_role_grant_from_a_mix(self):
    new_domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
    new_project = unit.new_project_ref(domain_id=new_domain['id'])
    PROVIDERS.resource_api.create_project(new_project['id'], new_project)
    new_group = unit.new_group_ref(domain_id=new_domain['id'])
    new_group = PROVIDERS.identity_api.create_group(new_group)
    new_group2 = unit.new_group_ref(domain_id=new_domain['id'])
    new_group2 = PROVIDERS.identity_api.create_group(new_group2)
    new_user = unit.new_user_ref(domain_id=new_domain['id'])
    new_user = PROVIDERS.identity_api.create_user(new_user)
    new_user2 = unit.new_user_ref(domain_id=new_domain['id'])
    new_user2 = PROVIDERS.identity_api.create_user(new_user2)
    PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], domain_id=new_domain['id'])
    self.assertEqual(0, len(roles_ref))
    PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
    PROVIDERS.assignment_api.create_grant(group_id=new_group2['id'], domain_id=new_domain['id'], role_id=self.role_admin['id'])
    PROVIDERS.assignment_api.create_grant(user_id=new_user2['id'], domain_id=new_domain['id'], role_id=self.role_admin['id'])
    PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], project_id=new_project['id'], role_id=self.role_admin['id'])
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], domain_id=new_domain['id'])
    self.assertDictEqual(self.role_member, roles_ref[0])
    PROVIDERS.assignment_api.delete_grant(group_id=new_group['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], domain_id=new_domain['id'])
    self.assertEqual(0, len(roles_ref))
    self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, group_id=new_group['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)