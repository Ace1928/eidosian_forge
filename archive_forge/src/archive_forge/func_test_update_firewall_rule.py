from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_update_firewall_rule(self):
    params = {'description': 'UpdatedDescription'}
    updated = self.mock_firewall_rule.copy()
    updated.update(params)
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_rules', self.firewall_rule_name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_rules', name=self.firewall_rule_name), json={'firewall_rules': [self.mock_firewall_rule]}), dict(method='PUT', uri=self._make_mock_url('firewall_rules', self.firewall_rule_id), json={'firewall_rule': updated}, validate=dict(json={'firewall_rule': params}))])
    self.assertDictEqual(updated, self.cloud.update_firewall_rule(self.firewall_rule_name, **params))
    self.assert_calls()