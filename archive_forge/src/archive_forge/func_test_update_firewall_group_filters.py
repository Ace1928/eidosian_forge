from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_update_firewall_group_filters(self):
    filters = {'project_id': self.mock_firewall_group['project_id']}
    params = {'description': 'updated again!'}
    updated_group = deepcopy(self.mock_returned_firewall_group)
    self.register_uris([dict(method='PUT', uri=self._make_mock_url('firewall_groups', self.firewall_group_id), json={'firewall_group': updated_group}, validate=dict(json={'firewall_group': params}))])
    with mock.patch.object(self.cloud.network, 'find_firewall_group', return_value=deepcopy(self.mock_firewall_group)):
        r = self.cloud.update_firewall_group(self.firewall_group_name, filters, **params)
        self.assertDictEqual(updated_group, r.to_dict())
        self.assert_calls()
        self.cloud.network.find_firewall_group.assert_called_once_with(self.firewall_group_name, ignore_missing=False, **filters)