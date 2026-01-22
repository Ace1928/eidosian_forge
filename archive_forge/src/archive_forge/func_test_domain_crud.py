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
def test_domain_crud(self):
    domain = unit.new_domain_ref()
    domain_ref = PROVIDERS.resource_api.create_domain(domain['id'], domain)
    self.assertDictEqual(domain, domain_ref)
    domain_ref = PROVIDERS.resource_api.get_domain(domain['id'])
    self.assertDictEqual(domain, domain_ref)
    domain['name'] = uuid.uuid4().hex
    domain_ref = PROVIDERS.resource_api.update_domain(domain['id'], domain)
    self.assertDictEqual(domain, domain_ref)
    domain_ref = PROVIDERS.resource_api.get_domain(domain['id'])
    self.assertDictEqual(domain, domain_ref)
    self.assertRaises(exception.ForbiddenNotSecurity, PROVIDERS.resource_api.delete_domain, domain_id=domain['id'])
    domain['enabled'] = False
    PROVIDERS.resource_api.update_domain(domain['id'], domain)
    PROVIDERS.resource_api.delete_domain(domain['id'])
    self.assertRaises(exception.DomainNotFound, PROVIDERS.resource_api.get_domain, domain['id'])