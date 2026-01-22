import os
import sys
import time
import unittest
from unittest.mock import Mock, patch
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.utils.py3 import u, httplib, assertRaisesRegex
from libcloud.compute.ssh import BaseSSHClient, SSHCommandTimeoutError, have_paramiko
from libcloud.compute.base import Node, NodeAuthPassword
from libcloud.test.secrets import RACKSPACE_PARAMS
from libcloud.compute.types import NodeState, LibcloudError, DeploymentError
from libcloud.compute.deployment import (
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.rackspace import RackspaceFirstGenNodeDriver as Rackspace
@unittest.skipIf(not have_paramiko, 'Skipping because paramiko is not available')
def test_ssh_client_connect_immediately_throws_on_fatal_execption(self):
    from paramiko.ssh_exception import SSHException, PasswordRequiredException
    mock_ssh_client = Mock()
    mock_ssh_client.connect = Mock()
    mock_ssh_client.connect.side_effect = IOError('bam')
    mock_exceptions = [SSHException('Invalid or unsupported key type'), PasswordRequiredException('private key file is encrypted'), SSHException('OpenSSH private key file checkints do not match')]
    for mock_exception in mock_exceptions:
        mock_ssh_client.connect = Mock(side_effect=mock_exception)
        assertRaisesRegex(self, mock_exception.__class__, str(mock_exception), self.driver._ssh_client_connect, ssh_client=mock_ssh_client, wait_period=0.1, timeout=0.2)