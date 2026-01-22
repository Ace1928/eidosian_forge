from unittest import mock
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware.tests import base
def test_vim_fault_exception_with_cause_and_details(self):
    vfe = exceptions.VimFaultException([ValueError('example')], 'MyMessage', 'FooBar', {'foo': 'bar'})
    string = str(vfe)
    self.assertIn(string, ["MyMessage\nCause: FooBar\nFaults: [ValueError('example',)]\nDetails: {'foo': 'bar'}", "MyMessage\nCause: FooBar\nFaults: [ValueError('example')]\nDetails: {'foo': 'bar'}"])