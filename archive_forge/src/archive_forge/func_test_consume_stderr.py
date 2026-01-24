import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_consume_stderr(self):
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu'}
    client = ParamikoSSHClient(**conn_params)
    client.CHUNK_SIZE = 1024
    chan = Mock()
    chan.recv_stderr_ready.side_effect = [True, True, False]
    chan.recv_stderr.side_effect = ['123', '456']
    stderr = client._consume_stderr(chan).getvalue()
    self.assertEqual(u('123456'), stderr)
    self.assertEqual(len(stderr), 6)
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu'}
    client = ParamikoSSHClient(**conn_params)
    client.CHUNK_SIZE = 1024
    chan = Mock()
    chan.recv_stderr_ready.side_effect = [True, True, False]
    chan.recv_stderr.side_effect = ['987', '6543210']
    stderr = client._consume_stderr(chan).getvalue()
    self.assertEqual(u('9892843210'), stderr)
    self.assertEqual(len(stderr), 10)