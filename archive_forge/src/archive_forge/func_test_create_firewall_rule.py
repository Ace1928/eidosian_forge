from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_create_firewall_rule(self):
    passed_attrs = self._mock_firewall_rule_attrs.copy()
    del passed_attrs['id']
    self.register_uris([dict(method='POST', uri=self._make_mock_url('firewall_rules'), json={'firewall_rule': self.mock_firewall_rule.copy()})])
    r = self.cloud.create_firewall_rule(**passed_attrs)
    self.assertDictEqual(self.mock_firewall_rule, r.to_dict())
    self.assert_calls()