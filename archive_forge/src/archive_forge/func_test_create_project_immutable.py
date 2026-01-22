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
def test_create_project_immutable(self):
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project['options'][ro_opt.IMMUTABLE_OPT.option_name] = True
    p_created = PROVIDERS.resource_api.create_project(project['id'], project)
    project_via_manager = PROVIDERS.resource_api.get_project(project['id'])
    self.assertTrue('options' in p_created)
    self.assertTrue('options' in project_via_manager)
    self.assertTrue(project_via_manager['options'][ro_opt.IMMUTABLE_OPT.option_name])
    self.assertTrue(p_created['options'][ro_opt.IMMUTABLE_OPT.option_name])