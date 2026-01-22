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
def test_create_domain_config_including_sensitive_option(self):
    config = {'ldap': {'url': uuid.uuid4().hex, 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}}
    PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
    res = PROVIDERS.domain_config_api.get_config(self.domain['id'])
    config_whitelisted = copy.deepcopy(config)
    config_whitelisted['ldap'].pop('password')
    self.assertEqual(config_whitelisted, res)
    res = PROVIDERS.domain_config_api.driver.get_config_option(self.domain['id'], 'ldap', 'password', sensitive=True)
    self.assertEqual(config['ldap']['password'], res['value'])
    res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
    self.assertEqual(config, res)