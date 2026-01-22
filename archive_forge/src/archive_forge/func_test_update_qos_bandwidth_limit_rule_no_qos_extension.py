import copy
from openstack import exceptions
from openstack.network.v2 import qos_bandwidth_limit_rule
from openstack.tests.unit import base
def test_update_qos_bandwidth_limit_rule_no_qos_extension(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': []})])
    self.assertRaises(exceptions.SDKException, self.cloud.update_qos_bandwidth_limit_rule, self.policy_id, self.rule_id, max_kbps=2000)
    self.assert_calls()