from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_system_grants_for_user(self):
    user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user_id = PROVIDERS.identity_api.create_user(user_ref)['id']
    first_role = self._create_role()
    second_role = self._create_role()
    PROVIDERS.assignment_api.create_system_grant_for_user(user_id, first_role['id'])
    system_roles = PROVIDERS.assignment_api.list_system_grants_for_user(user_id)
    self.assertEqual(len(system_roles), 1)
    PROVIDERS.assignment_api.create_system_grant_for_user(user_id, second_role['id'])
    system_roles = PROVIDERS.assignment_api.list_system_grants_for_user(user_id)
    self.assertEqual(len(system_roles), 2)