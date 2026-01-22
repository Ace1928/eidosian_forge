from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_and_remove_role_grant_by_user_and_cross_domain(self):
    user1_domain1_role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(user1_domain1_role['id'], user1_domain1_role)
    user1_domain2_role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(user1_domain2_role['id'], user1_domain2_role)
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    domain2 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
    user1 = unit.new_user_ref(domain_id=domain1['id'])
    user1 = PROVIDERS.identity_api.create_user(user1)
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain1['id'])
    self.assertEqual(0, len(roles_ref))
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain2['id'])
    self.assertEqual(0, len(roles_ref))
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=user1_domain1_role['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain2['id'], role_id=user1_domain2_role['id'])
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain1['id'])
    self.assertDictEqual(user1_domain1_role, roles_ref[0])
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain2['id'])
    self.assertDictEqual(user1_domain2_role, roles_ref[0])
    PROVIDERS.assignment_api.delete_grant(user_id=user1['id'], domain_id=domain2['id'], role_id=user1_domain2_role['id'])
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain2['id'])
    self.assertEqual(0, len(roles_ref))
    self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, user_id=user1['id'], domain_id=domain2['id'], role_id=user1_domain2_role['id'])