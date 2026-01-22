from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_get_vendor_from_path_win32_plink(self):
    manager = TestSSHVendorManager()
    manager.set_ssh_version_string('plink: Release 0.60')
    plink_path = 'C:/Program Files/PuTTY/plink.exe'
    self.overrideEnv('BRZ_SSH', plink_path)
    vendor = manager.get_vendor()
    self.assertIsInstance(vendor, PLinkSubprocessVendor)
    args = vendor._get_vendor_specific_argv('user', 'host', 22, ['bzr'])
    self.assertEqual(args[0], plink_path)