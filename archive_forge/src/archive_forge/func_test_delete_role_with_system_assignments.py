from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_delete_role_with_system_assignments(self):
    role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role['id'], role)
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    user = unit.new_user_ref(domain_id=domain['id'])
    user = PROVIDERS.identity_api.create_user(user)
    PROVIDERS.assignment_api.create_system_grant_for_user(user['id'], role['id'])
    PROVIDERS.role_api.delete_role(role['id'])
    system_roles = PROVIDERS.assignment_api.list_role_assignments(role_id=role['id'])
    self.assertEqual(len(system_roles), 0)