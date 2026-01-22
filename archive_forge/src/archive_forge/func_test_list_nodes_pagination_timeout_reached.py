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
@mock.patch('libcloud.compute.drivers.azure_arm.LIST_NODES_PAGINATION_TIMEOUT', 1)
def test_list_nodes_pagination_timeout_reached(self, fps_mock):
    AzureMockHttp.type = 'PAGINATION_INFINITE_LOOP'
    nodes = self.driver.list_nodes()
    self.assertTrue(len(nodes) >= 1)