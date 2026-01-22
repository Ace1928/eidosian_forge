from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_role_by_user_and_project(self):
    roles_ref = PROVIDERS.assignment_api.get_roles_for_user_and_project(self.user_foo['id'], self.project_bar['id'])
    self.assertNotIn(self.role_admin['id'], roles_ref)
    PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_foo['id'], self.project_bar['id'], self.role_admin['id'])
    roles_ref = PROVIDERS.assignment_api.get_roles_for_user_and_project(self.user_foo['id'], self.project_bar['id'])
    self.assertIn(self.role_admin['id'], roles_ref)
    self.assertNotIn(default_fixtures.MEMBER_ROLE_ID, roles_ref)
    PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_foo['id'], self.project_bar['id'], default_fixtures.MEMBER_ROLE_ID)
    roles_ref = PROVIDERS.assignment_api.get_roles_for_user_and_project(self.user_foo['id'], self.project_bar['id'])
    self.assertIn(self.role_admin['id'], roles_ref)
    self.assertIn(default_fixtures.MEMBER_ROLE_ID, roles_ref)