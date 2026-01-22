import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_delete_script(self):
    """
        Provide a basic test with 'delete' action.
        """
    mock = self.ssh_cli
    sd = '/root/random_script.sh'
    mock.connect()
    mock.delete(sd)
    mock.client.open_sftp().unlink.assert_called_with(sd)
    self.assertLogMsg('Deleting file')
    mock.close()
    self.assertLogMsg('Closing server connection')