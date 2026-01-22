import copy
from openstack import exceptions
from openstack.network.v2 import qos_minimum_bandwidth_rule
from openstack.tests.unit import base
def test_update_qos_minimum_bandwidth_rule(self):
    expected_rule = copy.copy(self.mock_rule)
    expected_rule['min_kbps'] = self.rule_min_kbps + 100
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': self.enabled_neutron_extensions}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id]), json=self.mock_policy), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id, 'minimum_bandwidth_rules', self.rule_id]), json={'minimum_bandwidth_rule': self.mock_rule}), dict(method='PUT', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id, 'minimum_bandwidth_rules', self.rule_id]), json={'minimum_bandwidth_rule': expected_rule}, validate=dict(json={'minimum_bandwidth_rule': {'min_kbps': self.rule_min_kbps + 100}}))])
    rule = self.cloud.update_qos_minimum_bandwidth_rule(self.policy_id, self.rule_id, min_kbps=self.rule_min_kbps + 100)
    self._compare_rules(expected_rule, rule)
    self.assert_calls()