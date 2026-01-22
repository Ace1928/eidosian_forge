import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_get_sftp_client(self):
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu'}
    client = ParamikoSSHClient(**conn_params)
    mock_client = Mock()
    mock_sft_client = Mock()
    mock_client.open_sftp.return_value = mock_sft_client
    client.client = mock_client
    self.assertEqual(mock_client.open_sftp.call_count, 0)
    self.assertEqual(client._get_sftp_client(), mock_sft_client)
    self.assertEqual(mock_client.open_sftp.call_count, 1)
    mock_client = Mock()
    mock_sft_client = Mock()
    client.client = mock_client
    client.sftp_client = mock_sft_client
    self.assertEqual(mock_client.open_sftp.call_count, 0)
    self.assertEqual(client._get_sftp_client(), mock_sft_client)
    self.assertEqual(mock_client.open_sftp.call_count, 0)
    mock_client = Mock()
    mock_sftp_client = Mock()
    client.client = mock_client
    client.sftp_client = mock_sftp_client
    mock_sftp_client.listdir.side_effect = OSError('Socket is closed')
    self.assertEqual(mock_client.open_sftp.call_count, 0)
    sftp_client = client._get_sftp_client()
    self.assertTrue(sftp_client != mock_sft_client)
    self.assertTrue(sftp_client)
    self.assertTrue(client._get_sftp_client())
    self.assertEqual(mock_client.open_sftp.call_count, 1)
    mock_client = Mock()
    mock_sftp_client = Mock()
    client.client = mock_client
    client.sftp_client = mock_sftp_client
    mock_sftp_client.listdir.side_effect = Exception('Fatal exception')
    self.assertEqual(mock_client.open_sftp.call_count, 0)
    self.assertRaisesRegex(Exception, 'Fatal exception', client._get_sftp_client)