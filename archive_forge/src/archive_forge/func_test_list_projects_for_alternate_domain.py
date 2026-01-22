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
@unit.skip_if_no_multiple_domains_support
def test_list_projects_for_alternate_domain(self):
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    project1 = unit.new_project_ref(domain_id=domain1['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    project2 = unit.new_project_ref(domain_id=domain1['id'])
    PROVIDERS.resource_api.create_project(project2['id'], project2)
    project_ids = [x['id'] for x in PROVIDERS.resource_api.list_projects_in_domain(domain1['id'])]
    self.assertEqual(2, len(project_ids))
    self.assertIn(project1['id'], project_ids)
    self.assertIn(project2['id'], project_ids)