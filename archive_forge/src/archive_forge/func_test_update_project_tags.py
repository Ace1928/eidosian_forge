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
def test_update_project_tags(self):
    project, tags = self._create_project_and_tags(num_of_tags=2)
    project_tag_ref = PROVIDERS.resource_api.list_project_tags(project['id'])
    self.assertEqual(len(project_tag_ref), 2)
    tags = ['one']
    PROVIDERS.resource_api.update_project_tags(project['id'], tags)
    project_tag_ref = PROVIDERS.resource_api.list_project_tags(project['id'])
    self.assertEqual(len(project_tag_ref), 1)