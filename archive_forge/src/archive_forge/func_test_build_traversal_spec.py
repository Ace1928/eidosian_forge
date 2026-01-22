import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_build_traversal_spec(self):
    client_factory = mock.Mock()
    sel_spec = mock.Mock()
    traversal_spec = vim_util.build_traversal_spec(client_factory, 'dc_to_hf', 'Datacenter', 'hostFolder', False, [sel_spec])
    self.assertEqual('dc_to_hf', traversal_spec.name)
    self.assertEqual('hostFolder', traversal_spec.path)
    self.assertEqual([sel_spec], traversal_spec.selectSet)
    self.assertFalse(traversal_spec.skip)
    self.assertEqual('Datacenter', traversal_spec.type)