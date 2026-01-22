from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_remove_user_role_not_assigned(self):
    self.assertRaises(exception.RoleNotFound, PROVIDERS.assignment_api.remove_role_from_user_and_project, project_id=self.project_bar['id'], user_id=self.user_two['id'], role_id=self.role_other['id'])