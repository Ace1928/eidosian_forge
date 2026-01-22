from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_roles_for_user_and_project_returns_not_found(self):
    self.assertRaises(exception.UserNotFound, PROVIDERS.assignment_api.get_roles_for_user_and_project, uuid.uuid4().hex, self.project_bar['id'])
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.assignment_api.get_roles_for_user_and_project, self.user_foo['id'], uuid.uuid4().hex)