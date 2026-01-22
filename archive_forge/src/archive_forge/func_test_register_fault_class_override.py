from unittest import mock
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware.tests import base
def test_register_fault_class_override(self):
    exc = self._create_subclass_exception()
    exceptions.register_fault_class(exceptions.ALREADY_EXISTS, exc)
    self.assertEqual(exc, exceptions.get_fault_class(exceptions.ALREADY_EXISTS))