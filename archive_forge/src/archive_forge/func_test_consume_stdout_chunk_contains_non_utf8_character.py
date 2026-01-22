import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_consume_stdout_chunk_contains_non_utf8_character(self):
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu'}
    client = ParamikoSSHClient(**conn_params)
    client.CHUNK_SIZE = 1
    chan = Mock()
    chan.recv_ready.side_effect = [True, True, True, False]
    chan.recv.side_effect = ['🤦'.encode('utf-32'), 'a', 'b']
    stdout = client._consume_stdout(chan).getvalue()
    if sys.byteorder == 'little':
        self.assertEqual('\x00\x00&\x01\x00ab', stdout)
    else:
        self.assertEqual('\x00\x00\x00\x01&ab', stdout)
    self.assertEqual(len(stdout), 7)