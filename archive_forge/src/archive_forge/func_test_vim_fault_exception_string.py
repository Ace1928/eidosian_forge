from unittest import mock
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware.tests import base
def test_vim_fault_exception_string(self):
    self.assertRaises(ValueError, exceptions.VimFaultException, 'bad', ValueError('argument'))