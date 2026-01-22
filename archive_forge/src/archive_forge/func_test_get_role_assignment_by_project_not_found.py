from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_role_assignment_by_project_not_found(self):
    self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_grant_role_id, user_id=self.user_foo['id'], project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
    self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_grant_role_id, group_id=uuid.uuid4().hex, project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)