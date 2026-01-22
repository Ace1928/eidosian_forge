from breezy import config
from breezy.errors import SSHVendorNotFound, UnknownSSH
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.transport.ssh import (LSHSubprocessVendor, OpenSSHSubprocessVendor,
def test_openssh_command_strange_hostname(self):
    vendor = SSHCorpSubprocessVendor()
    self.assertRaises(StrangeHostname, vendor._get_vendor_specific_argv, 'user', '-oProxyCommand=host', 100, command=['bzr'])