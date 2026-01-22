from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_lsh_command_arguments(self):
    vendor = LSHSubprocessVendor()
    self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, command=['bzr']), ['lsh', '-p', '100', '-l', 'user', 'host', 'bzr'])