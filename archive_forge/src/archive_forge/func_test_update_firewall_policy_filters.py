from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_update_firewall_policy_filters(self):
    filters = {'project_id': self.mock_firewall_policy['project_id']}
    params = {'description': 'updated!'}
    updated_policy = deepcopy(self.mock_firewall_policy)
    updated_policy.update(params)
    self.register_uris([dict(method='PUT', uri=self._make_mock_url('firewall_policies', self.firewall_policy_id), json={'firewall_policy': updated_policy}, validate=dict(json={'firewall_policy': params}))])
    with mock.patch.object(self.cloud.network, 'find_firewall_policy', return_value=deepcopy(self.mock_firewall_policy)):
        self.assertDictEqual(updated_policy, self.cloud.update_firewall_policy(self.firewall_policy_name, filters, **params))
        self.assert_calls()
        self.cloud.network.find_firewall_policy.assert_called_once_with(self.firewall_policy_name, ignore_missing=False, **filters)