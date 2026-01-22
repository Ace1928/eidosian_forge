from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_create_firewall_group_compact(self):
    """
        Tests firewall group creation without policies or ports
        """
    firewall_group = deepcopy(self._mock_firewall_group_attrs)
    del firewall_group['ports']
    del firewall_group['egress_firewall_policy']
    del firewall_group['ingress_firewall_policy']
    created_firewall = deepcopy(firewall_group)
    created_firewall.update(egress_firewall_policy_id=None, ingress_firewall_policy_id=None, ports=[])
    del firewall_group['id']
    self.register_uris([dict(method='POST', uri=self._make_mock_url('firewall_groups'), json={'firewall_group': created_firewall}, validate=dict(json={'firewall_group': firewall_group}))])
    r = self.cloud.create_firewall_group(**firewall_group)
    self.assertDictEqual(FirewallGroup(connection=self.cloud, **created_firewall).to_dict(), r.to_dict())
    self.assert_calls()