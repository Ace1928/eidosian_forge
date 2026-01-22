import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_get_base_ssh_command(self):
    client1 = ShellOutSSHClient(hostname='localhost', username='root')
    client2 = ShellOutSSHClient(hostname='localhost', username='root', key='/home/my.key')
    client3 = ShellOutSSHClient(hostname='localhost', username='root', key='/home/my.key', timeout=5)
    cmd1 = client1._get_base_ssh_command()
    cmd2 = client2._get_base_ssh_command()
    cmd3 = client3._get_base_ssh_command()
    self.assertEqual(cmd1, ['ssh', 'root@localhost'])
    self.assertEqual(cmd2, ['ssh', '-i', '/home/my.key', 'root@localhost'])
    self.assertEqual(cmd3, ['ssh', '-i', '/home/my.key', '-oConnectTimeout=5', 'root@localhost'])