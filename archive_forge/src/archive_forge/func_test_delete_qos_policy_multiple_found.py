import copy
from openstack import exceptions
from openstack.network.v2 import qos_policy as _policy
from openstack.tests.unit import base
def test_delete_qos_policy_multiple_found(self):
    policy1 = dict(id='123', name=self.policy_name)
    policy2 = dict(id='456', name=self.policy_name)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': self.enabled_neutron_extensions}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies'], qs_elements=['name=%s' % self.policy_name]), json={'policies': [policy1, policy2]})])
    self.assertRaises(exceptions.SDKException, self.cloud.delete_qos_policy, self.policy_name)
    self.assert_calls()