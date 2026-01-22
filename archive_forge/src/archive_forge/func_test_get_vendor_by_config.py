from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_get_vendor_by_config(self):
    manager = TestSSHVendorManager()
    self.overrideEnv('BRZ_SSH', None)
    self.assertRaises(SSHVendorNotFound, manager.get_vendor)
    config.GlobalStack().set('ssh', 'vendor')
    self.assertRaises(UnknownSSH, manager.get_vendor)
    vendor = object()
    manager.register_vendor('vendor', vendor)
    self.assertIs(manager.get_vendor(), vendor)