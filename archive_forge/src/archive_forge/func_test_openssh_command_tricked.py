from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_openssh_command_tricked(self):
    vendor = OpenSSHSubprocessVendor()
    self.assertEqual(vendor._get_vendor_specific_argv('user', '-oProxyCommand=blah', 100, command=['bzr']), ['ssh', '-oForwardX11=no', '-oForwardAgent=no', '-oClearAllForwardings=yes', '-oNoHostAuthenticationForLocalhost=yes', '-p', '100', '-l', 'user', '--', '-oProxyCommand=blah', 'bzr'])