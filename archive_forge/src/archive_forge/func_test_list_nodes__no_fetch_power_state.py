import sys
import json
import functools
from datetime import datetime
from unittest import mock
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import httplib, parse_qs, urlparse, urlunquote
from libcloud.common.types import LibcloudError
from libcloud.compute.base import NodeSize, NodeLocation, StorageVolume, VolumeSnapshot
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.azure_arm import (
@mock.patch('libcloud.compute.drivers.azure_arm.AzureNodeDriver._fetch_power_state', return_value=NodeState.UPDATING)
def test_list_nodes__no_fetch_power_state(self, fps_mock):
    nodes = self.driver.list_nodes(ex_fetch_power_state=False)
    self.assertEqual(len(nodes), 2)
    self.assertEqual(nodes[0].name, 'test-node-1')
    self.assertNotEqual(nodes[0].state, NodeState.UPDATING)
    self.assertEqual(nodes[0].private_ips, ['10.0.0.1'])
    self.assertEqual(nodes[0].public_ips, [])
    self.assertEqual(nodes[1].name, 'test-node-2')
    self.assertNotEqual(nodes[1].state, NodeState.UPDATING)
    self.assertEqual(nodes[1].private_ips, ['10.0.0.2'])
    self.assertEqual(nodes[1].public_ips, [])
    fps_mock.assert_not_called()