import collections
from unittest import mock
from oslo_vmware import dvs_util
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_trunk_vlan_spec(self):
    session = mock.Mock()
    spec = dvs_util.get_trunk_vlan_spec(session, start=1, end=2)
    self.assertEqual(1, spec.vlanId.start)
    self.assertEqual(2, spec.vlanId.end)