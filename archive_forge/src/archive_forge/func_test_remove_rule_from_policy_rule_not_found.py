from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_remove_rule_from_policy_rule_not_found(self):
    retrieved_policy = deepcopy(self.mock_firewall_policy)
    rule = FirewallRule(**TestFirewallRule._mock_firewall_rule_attrs)
    retrieved_policy['firewall_rules'][0] = rule['id']
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_policies', self.firewall_policy_id), json={'firewall_policy': retrieved_policy}), dict(method='GET', uri=self._make_mock_url('firewall_rules', rule['name']), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_rules', name=rule['name']), json={'firewall_rules': []})])
    r = self.cloud.remove_rule_from_policy(self.firewall_policy_id, rule['name'])
    self.assertDictEqual(retrieved_policy, r.to_dict())
    self.assert_calls()