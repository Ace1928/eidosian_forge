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
def test_update_invalid_partial_domain_config(self):
    config = {'ldap': {'url': uuid.uuid4().hex, 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
    self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.update_config, self.domain['id'], config, group='ldap')
    self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.update_config, self.domain['id'], config['ldap'], group='ldap', option='url')
    config = {'ldap': {'user_tree_dn': uuid.uuid4().hex}}
    self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.update_config, self.domain['id'], config, group='identity')
    self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.update_config, self.domain['id'], config['ldap'], group='ldap', option='url')
    config = {'ldap': {'user_tree_dn': uuid.uuid4().hex}}
    PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
    config_wrong_group = {'identity': {'driver': uuid.uuid4().hex}}
    self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.update_config, self.domain['id'], config_wrong_group, group='identity')
    config_wrong_option = {'url': uuid.uuid4().hex}
    self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.update_config, self.domain['id'], config_wrong_option, group='ldap', option='url')
    bad_group = uuid.uuid4().hex
    config = {bad_group: {'user': uuid.uuid4().hex}}
    self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.update_config, self.domain['id'], config, group=bad_group, option='user')
    bad_option = uuid.uuid4().hex
    config = {'ldap': {bad_option: uuid.uuid4().hex}}
    self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.update_config, self.domain['id'], config, group='ldap', option=bad_option)