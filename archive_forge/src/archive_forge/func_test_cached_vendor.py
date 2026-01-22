from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_cached_vendor(self):
    manager = TestSSHVendorManager()
    self.overrideEnv('BRZ_SSH', None)
    self.assertRaises(SSHVendorNotFound, manager.get_vendor)
    vendor = object()
    manager.register_vendor('vendor', vendor)
    self.assertRaises(SSHVendorNotFound, manager.get_vendor)
    self.overrideEnv('BRZ_SSH', 'vendor')
    self.assertIs(manager.get_vendor(), vendor)
    self.overrideEnv('BRZ_SSH', None)
    self.assertIs(manager.get_vendor(), vendor)
    manager.clear_cache()
    self.overrideEnv('BRZ_SSH', None)
    self.assertRaises(SSHVendorNotFound, manager.get_vendor)