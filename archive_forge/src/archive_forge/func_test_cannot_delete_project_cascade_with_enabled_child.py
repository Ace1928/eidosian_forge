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
def test_cannot_delete_project_cascade_with_enabled_child(self):
    projects_hierarchy = self._create_projects_hierarchy(hierarchy_size=3)
    root_project = projects_hierarchy[0]
    project1 = projects_hierarchy[1]
    project2 = projects_hierarchy[2]
    project2['enabled'] = False
    PROVIDERS.resource_api.update_project(project2['id'], project2)
    self.assertRaises(exception.ForbiddenNotSecurity, PROVIDERS.resource_api.delete_project, root_project['id'], cascade=True)
    PROVIDERS.resource_api.get_project(root_project['id'])
    PROVIDERS.resource_api.get_project(project1['id'])
    PROVIDERS.resource_api.get_project(project2['id'])