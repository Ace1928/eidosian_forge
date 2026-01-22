from unittest import mock
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware.tests import base
def test_get_fault_class(self):
    self.assertEqual(exceptions.AlreadyExistsException, exceptions.get_fault_class('AlreadyExists'))
    self.assertEqual(exceptions.CannotDeleteFileException, exceptions.get_fault_class('CannotDeleteFile'))
    self.assertEqual(exceptions.FileAlreadyExistsException, exceptions.get_fault_class('FileAlreadyExists'))
    self.assertEqual(exceptions.FileFaultException, exceptions.get_fault_class('FileFault'))
    self.assertEqual(exceptions.FileLockedException, exceptions.get_fault_class('FileLocked'))
    self.assertEqual(exceptions.FileNotFoundException, exceptions.get_fault_class('FileNotFound'))
    self.assertEqual(exceptions.InvalidPowerStateException, exceptions.get_fault_class('InvalidPowerState'))
    self.assertEqual(exceptions.InvalidPropertyException, exceptions.get_fault_class('InvalidProperty'))
    self.assertEqual(exceptions.NoPermissionException, exceptions.get_fault_class('NoPermission'))
    self.assertEqual(exceptions.NotAuthenticatedException, exceptions.get_fault_class('NotAuthenticated'))
    self.assertEqual(exceptions.TaskInProgress, exceptions.get_fault_class('TaskInProgress'))
    self.assertEqual(exceptions.DuplicateName, exceptions.get_fault_class('DuplicateName'))
    self.assertEqual(exceptions.NoDiskSpaceException, exceptions.get_fault_class('NoDiskSpace'))
    self.assertEqual(exceptions.ToolsUnavailableException, exceptions.get_fault_class('ToolsUnavailable'))
    self.assertEqual(exceptions.ManagedObjectNotFoundException, exceptions.get_fault_class('ManagedObjectNotFound'))
    self.assertIsNone(exceptions.get_fault_class('NotAFile'))