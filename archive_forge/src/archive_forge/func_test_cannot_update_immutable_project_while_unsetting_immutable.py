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
def test_cannot_update_immutable_project_while_unsetting_immutable(self):
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project['options'][ro_opt.IMMUTABLE_OPT.option_name] = True
    PROVIDERS.resource_api.create_project(project['id'], project)
    update_project = {'name': uuid.uuid4().hex, 'options': {ro_opt.IMMUTABLE_OPT.option_name: True}}
    self.assertRaises(exception.ResourceUpdateForbidden, PROVIDERS.resource_api.update_project, project['id'], update_project)