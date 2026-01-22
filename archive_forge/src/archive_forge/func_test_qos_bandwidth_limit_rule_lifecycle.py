from openstack import exceptions
from openstack.tests.functional import base
def test_qos_bandwidth_limit_rule_lifecycle(self):
    max_kbps = 1500
    max_burst_kbps = 500
    updated_max_kbps = 2000
    rule = self.operator_cloud.create_qos_bandwidth_limit_rule(self.policy['id'], max_kbps=max_kbps, max_burst_kbps=max_burst_kbps)
    self.assertIn('id', rule)
    self.assertEqual(max_kbps, rule['max_kbps'])
    self.assertEqual(max_burst_kbps, rule['max_burst_kbps'])
    updated_rule = self.operator_cloud.update_qos_bandwidth_limit_rule(self.policy['id'], rule['id'], max_kbps=updated_max_kbps)
    self.assertIn('id', updated_rule)
    self.assertEqual(updated_max_kbps, updated_rule['max_kbps'])
    self.assertEqual(max_burst_kbps, updated_rule['max_burst_kbps'])
    policy_rules = self.operator_cloud.list_qos_bandwidth_limit_rules(self.policy['id'])
    self.assertEqual([updated_rule], policy_rules)
    self.operator_cloud.delete_qos_bandwidth_limit_rule(self.policy['id'], updated_rule['id'])
    policy_rules = self.operator_cloud.list_qos_bandwidth_limit_rules(self.policy['id'])
    self.assertEqual([], policy_rules)