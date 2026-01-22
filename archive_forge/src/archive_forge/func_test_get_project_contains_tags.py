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
def test_get_project_contains_tags(self):
    project, _ = self._create_project_and_tags()
    tag = uuid.uuid4().hex
    PROVIDERS.resource_api.create_project_tag(project['id'], tag)
    ref = PROVIDERS.resource_api.get_project(project['id'])
    self.assertIn(tag, ref['tags'])