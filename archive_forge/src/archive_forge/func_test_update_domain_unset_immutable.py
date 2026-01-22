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
def test_update_domain_unset_immutable(self):
    domain_id = uuid.uuid4().hex
    domain = {'name': uuid.uuid4().hex, 'id': domain_id, 'is_domain': True}
    PROVIDERS.resource_api.create_domain(domain_id, domain)
    domain_via_manager = PROVIDERS.resource_api.get_domain(domain_id)
    self.assertTrue('options' in domain_via_manager)
    self.assertFalse(ro_opt.IMMUTABLE_OPT.option_name in domain_via_manager['options'])
    update_domain = {'options': {ro_opt.IMMUTABLE_OPT.option_name: False}}
    d_updated = PROVIDERS.resource_api.update_domain(domain_id, update_domain)
    domain_via_manager = PROVIDERS.resource_api.get_domain(domain_id)
    self.assertTrue('options' in domain_via_manager)
    self.assertTrue('options' in d_updated)
    self.assertTrue(ro_opt.IMMUTABLE_OPT.option_name in domain_via_manager['options'])
    self.assertTrue(ro_opt.IMMUTABLE_OPT.option_name in d_updated['options'])
    self.assertFalse(d_updated['options'][ro_opt.IMMUTABLE_OPT.option_name])
    self.assertFalse(domain_via_manager['options'][ro_opt.IMMUTABLE_OPT.option_name])
    update_domain = {'name': uuid.uuid4().hex}
    d_updated = PROVIDERS.resource_api.update_domain(domain_id, update_domain)
    self.assertEqual(d_updated['name'], update_domain['name'])
    update_domain = {'options': {ro_opt.IMMUTABLE_OPT.option_name: None}}
    d_updated = PROVIDERS.resource_api.update_domain(domain_id, update_domain)
    domain_via_manager = PROVIDERS.resource_api.get_domain(domain_id)
    self.assertTrue('options' in d_updated)
    self.assertTrue('options' in domain_via_manager)
    self.assertFalse(ro_opt.IMMUTABLE_OPT.option_name in d_updated['options'])
    self.assertFalse(ro_opt.IMMUTABLE_OPT.option_name in domain_via_manager['options'])