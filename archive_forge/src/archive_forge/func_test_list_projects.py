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
def test_list_projects(self):
    project_refs = PROVIDERS.resource_api.list_projects()
    project_count = len(default_fixtures.PROJECTS) + self.domain_count
    self.assertEqual(project_count, len(project_refs))
    for project in default_fixtures.PROJECTS:
        self.assertIn(project, project_refs)