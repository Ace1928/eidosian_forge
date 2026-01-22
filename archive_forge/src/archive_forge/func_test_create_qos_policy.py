import copy
from openstack import exceptions
from openstack.network.v2 import qos_policy as _policy
from openstack.tests.unit import base
def test_create_qos_policy(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': self.enabled_neutron_extensions}), dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies']), json={'policy': self.mock_policy})])
    policy = self.cloud.create_qos_policy(name=self.policy_name, project_id=self.project_id)
    self._compare_policies(self.mock_policy, policy)
    self.assert_calls()