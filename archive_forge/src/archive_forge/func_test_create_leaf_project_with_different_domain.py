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
def test_create_leaf_project_with_different_domain(self):
    root_project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.resource_api.create_project(root_project['id'], root_project)
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    leaf_project = unit.new_project_ref(domain_id=domain['id'], parent_id=root_project['id'])
    self.assertRaises(exception.ValidationError, PROVIDERS.resource_api.create_project, leaf_project['id'], leaf_project)