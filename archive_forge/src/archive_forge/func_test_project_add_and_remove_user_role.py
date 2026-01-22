from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_project_add_and_remove_user_role(self):
    user_ids = PROVIDERS.assignment_api.list_user_ids_for_project(self.project_bar['id'])
    self.assertNotIn(self.user_two['id'], user_ids)
    PROVIDERS.assignment_api.add_role_to_user_and_project(project_id=self.project_bar['id'], user_id=self.user_two['id'], role_id=self.role_other['id'])
    user_ids = PROVIDERS.assignment_api.list_user_ids_for_project(self.project_bar['id'])
    self.assertIn(self.user_two['id'], user_ids)
    PROVIDERS.assignment_api.remove_role_from_user_and_project(project_id=self.project_bar['id'], user_id=self.user_two['id'], role_id=self.role_other['id'])
    user_ids = PROVIDERS.assignment_api.list_user_ids_for_project(self.project_bar['id'])
    self.assertNotIn(self.user_two['id'], user_ids)