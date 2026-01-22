import copy
from unittest import mock
import uuid
from oslo_config import cfg
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def test_delete_domain_deletes_configs(self):
    """Test domain deletion clears the domain configs."""
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    config = {'ldap': {'url': uuid.uuid4().hex, 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}}
    PROVIDERS.domain_config_api.create_config(domain['id'], config)
    domain['enabled'] = False
    PROVIDERS.resource_api.update_domain(domain['id'], domain)
    PROVIDERS.resource_api.delete_domain(domain['id'])
    self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.get_config, domain['id'])
    self.assertDictEqual({}, PROVIDERS.domain_config_api.get_config_with_sensitive_info(domain['id']))