import copy
from openstack import exceptions
from openstack.network.v2 import qos_dscp_marking_rule
from openstack.tests.unit import base
def test_get_qos_dscp_marking_rule(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': self.enabled_neutron_extensions}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies'], qs_elements=['name=%s' % self.policy_name]), json={'policies': [self.mock_policy]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id, 'dscp_marking_rules', self.rule_id]), json={'dscp_marking_rule': self.mock_rule})])
    r = self.cloud.get_qos_dscp_marking_rule(self.policy_name, self.rule_id)
    self._compare_rules(self.mock_rule, r)
    self.assert_calls()