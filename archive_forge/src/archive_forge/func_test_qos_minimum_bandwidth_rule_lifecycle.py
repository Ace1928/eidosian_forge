from openstack import exceptions
from openstack.tests.functional import base
def test_qos_minimum_bandwidth_rule_lifecycle(self):
    min_kbps = 1500
    updated_min_kbps = 2000
    rule = self.operator_cloud.create_qos_minimum_bandwidth_rule(self.policy['id'], min_kbps=min_kbps)
    self.assertIn('id', rule)
    self.assertEqual(min_kbps, rule['min_kbps'])
    updated_rule = self.operator_cloud.update_qos_minimum_bandwidth_rule(self.policy['id'], rule['id'], min_kbps=updated_min_kbps)
    self.assertIn('id', updated_rule)
    self.assertEqual(updated_min_kbps, updated_rule['min_kbps'])
    policy_rules = self.operator_cloud.list_qos_minimum_bandwidth_rules(self.policy['id'])
    self.assertEqual([updated_rule], policy_rules)
    self.operator_cloud.delete_qos_minimum_bandwidth_rule(self.policy['id'], updated_rule['id'])
    policy_rules = self.operator_cloud.list_qos_minimum_bandwidth_rules(self.policy['id'])
    self.assertEqual([], policy_rules)