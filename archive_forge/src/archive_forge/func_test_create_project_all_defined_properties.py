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
def test_create_project_all_defined_properties(self):
    project_id = uuid.uuid4().hex
    project = {'name': uuid.uuid4().hex, 'id': project_id, 'domain_id': default_fixtures.ROOT_DOMAIN['id']}
    parent_project = self.driver.create_project(project_id, project)
    project_id = uuid.uuid4().hex
    project = {'name': uuid.uuid4().hex, 'id': project_id, 'domain_id': default_fixtures.ROOT_DOMAIN['id'], 'description': uuid.uuid4().hex, 'enabled': True, 'parent_id': parent_project['id'], 'is_domain': True}
    self.driver.create_project(project_id, project)