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
@unit.skip_if_cache_disabled('resource')
@unit.skip_if_no_multiple_domains_support
def test_cache_layer_project_crud(self):
    domain = unit.new_domain_ref()
    project = unit.new_project_ref(domain_id=domain['id'])
    project_id = project['id']
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    PROVIDERS.resource_api.create_project(project_id, project)
    PROVIDERS.resource_api.get_project(project_id)
    updated_project = copy.deepcopy(project)
    updated_project['name'] = uuid.uuid4().hex
    PROVIDERS.resource_api.driver.update_project(project_id, updated_project)
    self.assertLessEqual(project.items(), PROVIDERS.resource_api.get_project(project_id).items())
    PROVIDERS.resource_api.get_project.invalidate(PROVIDERS.resource_api, project_id)
    self.assertLessEqual(updated_project.items(), PROVIDERS.resource_api.get_project(project_id).items())
    PROVIDERS.resource_api.update_project(project['id'], project)
    self.assertLessEqual(project.items(), PROVIDERS.resource_api.get_project(project_id).items())
    PROVIDERS.resource_api.driver.delete_project(project_id)
    self.assertLessEqual(project.items(), PROVIDERS.resource_api.get_project(project_id).items())
    PROVIDERS.resource_api.get_project.invalidate(PROVIDERS.resource_api, project_id)
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.get_project, project_id)
    PROVIDERS.resource_api.create_project(project_id, project)
    PROVIDERS.resource_api.get_project(project_id)
    PROVIDERS.resource_api.delete_project(project_id)
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.get_project, project_id)