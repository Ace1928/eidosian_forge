import collections
from unittest import mock
from oslo_vmware import dvs_util
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_dvs_moref(self):
    moref = dvs_util.get_dvs_moref('dvs-123')
    self.assertEqual('dvs-123', vim_util.get_moref_value(moref))
    self.assertEqual('VmwareDistributedVirtualSwitch', vim_util.get_moref_type(moref))