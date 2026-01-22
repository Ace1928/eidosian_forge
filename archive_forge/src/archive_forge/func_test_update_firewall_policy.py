from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_update_firewall_policy(self):
    lookup_rule = FirewallRule(connection=self.cloud, **TestFirewallRule._mock_firewall_rule_attrs).to_dict()
    params = {'firewall_rules': [lookup_rule['id']], 'description': 'updated!'}
    retrieved_policy = deepcopy(self.mock_firewall_policy)
    del retrieved_policy['firewall_rules'][0]
    updated_policy = deepcopy(self.mock_firewall_policy)
    updated_policy.update(params)
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_policies', self.firewall_policy_name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_policies', name=self.firewall_policy_name), json={'firewall_policies': [retrieved_policy]}), dict(method='GET', uri=self._make_mock_url('firewall_rules', lookup_rule['id']), json={'firewall_rule': lookup_rule}), dict(method='PUT', uri=self._make_mock_url('firewall_policies', self.firewall_policy_id), json={'firewall_policy': updated_policy}, validate=dict(json={'firewall_policy': params}))])
    self.assertDictEqual(updated_policy, self.cloud.update_firewall_policy(self.firewall_policy_name, **params))
    self.assert_calls()