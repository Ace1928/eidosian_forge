from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_delete_project_with_role_assignments(self):
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.resource_api.create_project(project['id'], project)
    PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_foo['id'], project['id'], default_fixtures.MEMBER_ROLE_ID)
    PROVIDERS.resource_api.delete_project(project['id'])
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.assignment_api.list_user_ids_for_project, project['id'])