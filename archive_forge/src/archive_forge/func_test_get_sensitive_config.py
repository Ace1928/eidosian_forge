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
def test_get_sensitive_config(self):
    config = {'ldap': {'url': uuid.uuid4().hex, 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
    res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
    self.assertEqual({}, res)
    PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
    res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
    self.assertEqual(config, res)