from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_list_firewall_policies(self):
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_policies'), json={'firewall_policies': [self.mock_firewall_policy.copy(), self.mock_firewall_policy.copy()]})])
    policy = FirewallPolicy(connection=self.cloud, **self.mock_firewall_policy)
    self.assertListEqual(self.cloud.list_firewall_policies(), [policy, policy])
    self.assert_calls()