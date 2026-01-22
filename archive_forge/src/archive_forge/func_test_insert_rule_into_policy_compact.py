from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_insert_rule_into_policy_compact(self):
    """
        Tests without insert_after and insert_before
        """
    rule = FirewallRule(**TestFirewallRule._mock_firewall_rule_attrs)
    retrieved_policy = deepcopy(self.mock_firewall_policy)
    retrieved_policy['firewall_rules'] = []
    updated_policy = deepcopy(retrieved_policy)
    updated_policy['firewall_rules'].append(rule['id'])
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_policies', self.firewall_policy_name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_policies', name=self.firewall_policy_name), json={'firewall_policies': [retrieved_policy]}), dict(method='GET', uri=self._make_mock_url('firewall_rules', rule['name']), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_rules', name=rule['name']), json={'firewall_rules': [rule]}), dict(method='PUT', uri=self._make_mock_url('firewall_policies', retrieved_policy['id'], 'insert_rule'), json=updated_policy, validate=dict(json={'firewall_rule_id': rule['id'], 'insert_after': None, 'insert_before': None}))])
    r = self.cloud.insert_rule_into_policy(self.firewall_policy_name, rule['name'])
    self.assertDictEqual(updated_policy, r.to_dict())
    self.assert_calls()