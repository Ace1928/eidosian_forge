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
def test_domain_name_case_sensitivity(self):
    domain_name = 'test_domain'
    ref = unit.new_domain_ref(name=domain_name)
    lower_case_domain = PROVIDERS.resource_api.create_domain(ref['id'], ref)
    ref['id'] = uuid.uuid4().hex
    ref['name'] = domain_name.upper()
    upper_case_domain = PROVIDERS.resource_api.create_domain(ref['id'], ref)
    lower_case_domain_ref = PROVIDERS.resource_api.get_domain_by_name(domain_name)
    self.assertDictEqual(lower_case_domain, lower_case_domain_ref)
    upper_case_domain_ref = PROVIDERS.resource_api.get_domain_by_name(domain_name.upper())
    self.assertDictEqual(upper_case_domain, upper_case_domain_ref)