import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_moref(self):
    moref = vim_util.get_moref('vm-0', 'VirtualMachine')
    self.assertEqual('vm-0', vim_util.get_moref_value(moref))
    self.assertEqual('VirtualMachine', vim_util.get_moref_type(moref))