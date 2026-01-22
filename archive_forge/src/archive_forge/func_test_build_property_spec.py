import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_build_property_spec(self):
    client_factory = mock.Mock()
    prop_spec = vim_util.build_property_spec(client_factory)
    self.assertFalse(prop_spec.all)
    self.assertEqual(['name'], prop_spec.pathSet)
    self.assertEqual('VirtualMachine', prop_spec.type)