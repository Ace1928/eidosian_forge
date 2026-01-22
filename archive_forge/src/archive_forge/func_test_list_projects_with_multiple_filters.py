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
def test_list_projects_with_multiple_filters(self):
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project = PROVIDERS.resource_api.create_project(project['id'], project)
    hints = driver_hints.Hints()
    hints.add_filter('name', project['name'])
    hints.add_filter('description', uuid.uuid4().hex)
    projects = PROVIDERS.resource_api.list_projects(hints)
    self.assertEqual([], projects)
    hints = driver_hints.Hints()
    hints.add_filter('name', project['name'])
    hints.add_filter('description', project['description'])
    projects = PROVIDERS.resource_api.list_projects(hints)
    self.assertEqual(1, len(projects))
    self.assertEqual(project, projects[0])