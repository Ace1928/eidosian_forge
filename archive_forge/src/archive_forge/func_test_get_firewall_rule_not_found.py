from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_get_firewall_rule_not_found(self):
    name = 'not_found'
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_rules', name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_rules', name=name), json={'firewall_rules': []})])
    self.assertIsNone(self.cloud.get_firewall_rule(name))
    self.assert_calls()