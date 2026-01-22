import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.resource.backends import sql as resource_sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import utils as test_utils
def test_delete_project_with_roles_clears_default_project_id(self):
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id, project_id=project['id'])
    PROVIDERS.resource_api.create_project(project['id'], project)
    user = PROVIDERS.identity_api.create_user(user)
    role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role['id'], role)
    PROVIDERS.assignment_api.create_grant(user_id=user['id'], project_id=project['id'], role_id=role['id'])
    PROVIDERS.resource_api.delete_project(project['id'])
    user = PROVIDERS.identity_api.get_user(user['id'])
    self.assertNotIn('default_project_id', user)