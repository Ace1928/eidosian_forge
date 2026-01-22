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
def test_update_project_unset_immutable(self):
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project['options'][ro_opt.IMMUTABLE_OPT.option_name] = True
    PROVIDERS.resource_api.create_project(project['id'], project)
    project_via_manager = PROVIDERS.resource_api.get_project(project['id'])
    self.assertTrue('options' in project_via_manager)
    self.assertTrue(project_via_manager['options'][ro_opt.IMMUTABLE_OPT.option_name])
    update_project = {'options': {ro_opt.IMMUTABLE_OPT.option_name: False}}
    PROVIDERS.resource_api.update_project(project['id'], update_project)
    project_via_manager = PROVIDERS.resource_api.get_project(project['id'])
    self.assertTrue('options' in project_via_manager)
    self.assertTrue(ro_opt.IMMUTABLE_OPT.option_name in project_via_manager['options'])
    self.assertFalse(project_via_manager['options'][ro_opt.IMMUTABLE_OPT.option_name])
    update_project = {'name': uuid.uuid4().hex}
    p_updated = PROVIDERS.resource_api.update_project(project['id'], update_project)
    self.assertEqual(p_updated['name'], update_project['name'])
    update_project = {'options': {ro_opt.IMMUTABLE_OPT.option_name: None}}
    p_updated = PROVIDERS.resource_api.update_project(project['id'], update_project)
    project_via_manager = PROVIDERS.resource_api.get_project(project['id'])
    self.assertTrue('options' in p_updated)
    self.assertTrue('options' in project_via_manager)
    self.assertFalse(ro_opt.IMMUTABLE_OPT.option_name in p_updated['options'])
    self.assertFalse(ro_opt.IMMUTABLE_OPT.option_name in project_via_manager['options'])