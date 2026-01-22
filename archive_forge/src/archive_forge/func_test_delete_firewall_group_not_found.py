from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_delete_firewall_group_not_found(self):
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_groups', self.firewall_group_name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_groups', name=self.firewall_group_name), json={'firewall_groups': []})])
    with mock.patch.object(self.cloud.log, 'debug'):
        self.assertFalse(self.cloud.delete_firewall_group(self.firewall_group_name))
        self.assert_calls()
        self.cloud.log.debug.assert_called_once()