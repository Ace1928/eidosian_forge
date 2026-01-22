from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_plink_subsystem_arguments(self):
    vendor = PLinkSubprocessVendor()
    self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, subsystem='sftp'), ['plink', '-x', '-a', '-ssh', '-2', '-batch', '-P', '100', '-l', 'user', '-s', 'host', 'sftp'])