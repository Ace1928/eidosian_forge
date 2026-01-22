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
def test_list_projects_for_domain(self):
    project_ids = [x['id'] for x in PROVIDERS.resource_api.list_projects_in_domain(CONF.identity.default_domain_id)]
    self.assertThat(project_ids, matchers.HasLength(len(default_fixtures.PROJECTS)))
    self.assertIn(self.project_bar['id'], project_ids)
    self.assertIn(self.project_baz['id'], project_ids)
    self.assertIn(self.project_mtu['id'], project_ids)
    self.assertIn(self.project_service['id'], project_ids)