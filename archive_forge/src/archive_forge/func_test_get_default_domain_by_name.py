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
def test_get_default_domain_by_name(self):
    domain_name = 'default'
    domain = unit.new_domain_ref(name=domain_name)
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    domain_ref = PROVIDERS.resource_api.get_domain_by_name(domain_name)
    self.assertEqual(domain, domain_ref)