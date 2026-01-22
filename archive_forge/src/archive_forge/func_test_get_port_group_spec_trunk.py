import collections
from unittest import mock
from oslo_vmware import dvs_util
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_port_group_spec_trunk(self):
    session = mock.Mock()
    spec = dvs_util.get_port_group_spec(session, 'pg', None, trunk_mode=True)
    self.assertEqual('pg', spec.name)
    self.assertEqual('ephemeral', spec.type)
    self.assertEqual(0, spec.defaultPortConfig.vlan.start)
    self.assertEqual(4094, spec.defaultPortConfig.vlan.end)