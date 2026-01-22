import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_prop_spec(self):
    client_factory = mock.Mock()
    prop_spec = vim_util.get_prop_spec(client_factory, 'VirtualMachine', ['test_path'])
    self.assertEqual(['test_path'], prop_spec.pathSet)
    self.assertEqual('VirtualMachine', prop_spec.type)