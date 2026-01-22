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
def test_run_deployment_script_ssh_command_timeout_fatal_exception(self):
    task = Mock()
    task.run = Mock()
    task.run.side_effect = SSHCommandTimeoutError('ls -la', 10)
    ssh_client = Mock()
    try:
        self.driver._run_deployment_script(task=task, node=self.node, ssh_client=ssh_client, max_tries=5)
    except SSHCommandTimeoutError as e:
        self.assertTrue(e.message.find("Command didn't finish") != -1)
    else:
        self.fail('Exception was not thrown')