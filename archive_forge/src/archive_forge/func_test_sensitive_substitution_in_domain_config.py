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
def test_sensitive_substitution_in_domain_config(self):
    config = {'ldap': {'url': 'my_url/%(password)s', 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
    PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
    res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
    expected_url = config['ldap']['url'] % {'password': config['ldap']['password']}
    self.assertEqual(expected_url, res['ldap']['url'])