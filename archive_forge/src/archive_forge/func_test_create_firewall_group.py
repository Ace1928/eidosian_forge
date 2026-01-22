from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_create_firewall_group(self):
    create_group_attrs = self._mock_firewall_group_attrs.copy()
    del create_group_attrs['id']
    posted_group_attrs = self._mock_returned_firewall_group_attrs.copy()
    del posted_group_attrs['egress_firewall_policy']
    del posted_group_attrs['ingress_firewall_policy']
    del posted_group_attrs['id']
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_policies', self.mock_egress_policy['name']), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_policies', name=self.mock_egress_policy['name']), json={'firewall_policies': [self.mock_egress_policy]}), dict(method='GET', uri=self._make_mock_url('firewall_policies', self.mock_ingress_policy['name']), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_policies', name=self.mock_ingress_policy['name']), json={'firewall_policies': [self.mock_ingress_policy]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports', self.mock_port['name']]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports'], qs_elements=['name=%s' % self.mock_port['name']]), json={'ports': [self.mock_port]}), dict(method='POST', uri=self._make_mock_url('firewall_groups'), json={'firewall_group': deepcopy(self.mock_returned_firewall_group)}, validate=dict(json={'firewall_group': posted_group_attrs}))])
    r = self.cloud.create_firewall_group(**create_group_attrs)
    self.assertDictEqual(self.mock_returned_firewall_group, r.to_dict())
    self.assert_calls()