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
def test_update_project_enable(self):
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.resource_api.create_project(project['id'], project)
    project_ref = PROVIDERS.resource_api.get_project(project['id'])
    self.assertTrue(project_ref['enabled'])
    project['enabled'] = False
    PROVIDERS.resource_api.update_project(project['id'], project)
    project_ref = PROVIDERS.resource_api.get_project(project['id'])
    self.assertEqual(project['enabled'], project_ref['enabled'])
    del project['enabled']
    PROVIDERS.resource_api.update_project(project['id'], project)
    project_ref = PROVIDERS.resource_api.get_project(project['id'])
    self.assertFalse(project_ref['enabled'])
    project['enabled'] = True
    PROVIDERS.resource_api.update_project(project['id'], project)
    project_ref = PROVIDERS.resource_api.get_project(project['id'])
    self.assertEqual(project['enabled'], project_ref['enabled'])
    del project['enabled']
    PROVIDERS.resource_api.update_project(project['id'], project)
    project_ref = PROVIDERS.resource_api.get_project(project['id'])
    self.assertTrue(project_ref['enabled'])