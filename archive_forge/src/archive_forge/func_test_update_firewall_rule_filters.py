from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_update_firewall_rule_filters(self):
    params = {'description': 'Updated!'}
    filters = {'project_id': self.mock_firewall_rule['project_id']}
    updated = self.mock_firewall_rule.copy()
    updated.update(params)
    updated_dict = self._mock_firewall_rule_attrs.copy()
    updated_dict.update(params)
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_rules', self.firewall_rule_name, **filters), json={'firewall_rule': self._mock_firewall_rule_attrs}), dict(method='PUT', uri=self._make_mock_url('firewall_rules', self.firewall_rule_id), json={'firewall_rule': updated_dict}, validate={'json': {'firewall_rule': params}})])
    updated_rule = self.cloud.update_firewall_rule(self.firewall_rule_name, filters, **params)
    self.assertDictEqual(updated, updated_rule)
    self.assert_calls()