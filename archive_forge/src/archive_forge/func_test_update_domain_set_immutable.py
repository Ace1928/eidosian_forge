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
def test_update_domain_set_immutable(self):
    domain_id = uuid.uuid4().hex
    domain = {'name': uuid.uuid4().hex, 'id': domain_id, 'is_domain': True}
    PROVIDERS.resource_api.create_domain(domain_id, domain)
    domain_via_manager = PROVIDERS.resource_api.get_domain(domain_id)
    self.assertTrue('options' in domain_via_manager)
    self.assertFalse(ro_opt.IMMUTABLE_OPT.option_name in domain_via_manager['options'])
    domain_update = {'options': {ro_opt.IMMUTABLE_OPT.option_name: True}}
    d_update = PROVIDERS.resource_api.update_domain(domain_id, domain_update)
    domain_via_manager = PROVIDERS.resource_api.get_domain(domain_id)
    self.assertTrue(ro_opt.IMMUTABLE_OPT.option_name in d_update['options'])
    self.assertTrue(d_update['options'][ro_opt.IMMUTABLE_OPT.option_name])
    self.assertTrue(ro_opt.IMMUTABLE_OPT.option_name in domain_via_manager['options'])
    self.assertTrue(domain_via_manager['options'][ro_opt.IMMUTABLE_OPT.option_name])