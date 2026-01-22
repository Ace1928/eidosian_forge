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
def test_script_deployment(self):
    sd1 = ScriptDeployment(script='foobar', delete=True)
    sd2 = ScriptDeployment(script='foobar', delete=False)
    sd3 = ScriptDeployment(script='foobar', delete=False, name='foobarname')
    sd4 = ScriptDeployment(script='foobar', delete=False, name='foobarname', timeout=10)
    self.assertTrue(sd1.name.find('deployment') != '1')
    self.assertEqual(sd3.name, 'foobarname')
    self.assertEqual(sd3.timeout, None)
    self.assertEqual(sd4.timeout, 10)
    self.assertEqual(self.node, sd1.run(node=self.node, client=MockClient(hostname='localhost')))
    self.assertEqual(self.node, sd2.run(node=self.node, client=MockClient(hostname='localhost')))
    self.assertEqual(self.node, sd3.run(node=self.node, client=MockClient(hostname='localhost')))
    assertRaisesRegex(self, ValueError, 'timeout', sd4.run, node=self.node, client=MockClient(hostname='localhost', throw_on_timeout=True))