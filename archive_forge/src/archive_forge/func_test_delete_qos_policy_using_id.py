import copy
from openstack import exceptions
from openstack.network.v2 import qos_policy as _policy
from openstack.tests.unit import base
def test_delete_qos_policy_using_id(self):
    policy1 = self.mock_policy
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': self.enabled_neutron_extensions}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', policy1['id']]), json=policy1), dict(method='DELETE', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id]), json={})])
    self.assertTrue(self.cloud.delete_qos_policy(policy1['id']))
    self.assert_calls()