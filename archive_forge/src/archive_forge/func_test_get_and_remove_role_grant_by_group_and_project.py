from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_and_remove_role_grant_by_group_and_project(self):
    new_domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
    new_group = unit.new_group_ref(domain_id=new_domain['id'])
    new_group = PROVIDERS.identity_api.create_group(new_group)
    new_user = unit.new_user_ref(domain_id=new_domain['id'])
    new_user = PROVIDERS.identity_api.create_user(new_user)
    PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], project_id=self.project_bar['id'])
    self.assertEqual(0, len(roles_ref))
    PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], project_id=self.project_bar['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], project_id=self.project_bar['id'])
    self.assertDictEqual(self.role_member, roles_ref[0])
    PROVIDERS.assignment_api.delete_grant(group_id=new_group['id'], project_id=self.project_bar['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
    roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], project_id=self.project_bar['id'])
    self.assertEqual(0, len(roles_ref))
    self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, group_id=new_group['id'], project_id=self.project_bar['id'], role_id=default_fixtures.MEMBER_ROLE_ID)