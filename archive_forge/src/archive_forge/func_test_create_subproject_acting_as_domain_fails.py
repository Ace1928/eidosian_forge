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
def test_create_subproject_acting_as_domain_fails(self):
    root_project = unit.new_project_ref(is_domain=True)
    PROVIDERS.resource_api.create_project(root_project['id'], root_project)
    sub_project = unit.new_project_ref(is_domain=True, parent_id=root_project['id'])
    self.assertRaises(exception.ValidationError, PROVIDERS.resource_api.create_project, sub_project['id'], sub_project)