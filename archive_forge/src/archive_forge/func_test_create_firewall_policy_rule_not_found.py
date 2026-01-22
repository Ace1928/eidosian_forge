from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_create_firewall_policy_rule_not_found(self):
    posted_policy = deepcopy(self._mock_firewall_policy_attrs)
    del posted_policy['id']
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_rules', posted_policy['firewall_rules'][0]), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_rules', name=posted_policy['firewall_rules'][0]), json={'firewall_rules': []})])
    with mock.patch.object(self.cloud.network, 'create_firewall_policy'):
        self.assertRaises(exceptions.ResourceNotFound, self.cloud.create_firewall_policy, **posted_policy)
        self.cloud.network.create_firewall_policy.assert_not_called()
        self.assert_calls()