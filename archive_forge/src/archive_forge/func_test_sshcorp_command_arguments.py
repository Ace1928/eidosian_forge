from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_sshcorp_command_arguments(self):
    vendor = SSHCorpSubprocessVendor()
    self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, command=['bzr']), ['ssh', '-x', '-p', '100', '-l', 'user', 'host', 'bzr'])