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
def test_update_network_profile(self):
    nics = self.driver.ex_list_nics()
    node = self.driver.list_nodes()[0]
    network_profile = node.extra['properties']['networkProfile']
    primary_nic_exists = False
    num_nics_before = len(network_profile['networkInterfaces'])
    for nic in network_profile['networkInterfaces']:
        if 'properties' in nic and nic['properties']['primary']:
            primary_nic_exists = True
    if not primary_nic_exists:
        network_profile['networkInterfaces'][0]['properties'] = {'primary': True}
    network_profile['networkInterfaces'].append({'id': nics[0].id})
    self.driver.ex_update_network_profile_of_node(node, network_profile)
    network_profile = node.extra['properties']['networkProfile']
    num_nics_after = len(network_profile['networkInterfaces'])
    self.assertEqual(num_nics_after, num_nics_before + 1)