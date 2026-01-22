from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_get_vendor_by_inspection_plink(self):
    manager = TestSSHVendorManager()
    self.overrideEnv('BRZ_SSH', None)
    self.assertRaises(SSHVendorNotFound, manager.get_vendor)
    manager.set_ssh_version_string('plink')
    self.assertRaises(SSHVendorNotFound, manager.get_vendor)