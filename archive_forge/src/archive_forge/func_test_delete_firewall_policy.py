from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_delete_firewall_policy(self):
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_policies', self.firewall_policy_name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_policies', name=self.firewall_policy_name), json={'firewall_policies': [self.mock_firewall_policy]}), dict(method='DELETE', uri=self._make_mock_url('firewall_policies', self.firewall_policy_id), json={}, status_code=204)])
    with mock.patch.object(self.cloud.log, 'debug'):
        self.assertTrue(self.cloud.delete_firewall_policy(self.firewall_policy_name))
        self.assert_calls()
        self.cloud.log.debug.assert_not_called()