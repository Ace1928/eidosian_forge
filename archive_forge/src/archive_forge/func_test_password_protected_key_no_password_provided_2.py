import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
@patch('paramiko.SSHClient', Mock)
def test_password_protected_key_no_password_provided_2(self):
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'misc', 'test_rsa_2048b_pass_foobar.key')
    with open(path) as fp:
        private_key = fp.read()
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu', 'key_material': private_key, 'password': 'invalid'}
    mock = ParamikoSSHClient(**conn_params)
    expected_msg = 'OpenSSH private key file checkints do not match'
    assertRaisesRegex(self, paramiko.ssh_exception.SSHException, expected_msg, mock.connect)