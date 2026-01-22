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
def test_create_volume__with_snapshot(self):
    location = self.driver.list_locations()[0]
    snap_id = '/subscriptions/99999999-9999-9999-9999-999999999999/resourceGroups/000000/providers/Microsoft.Compute/snapshots/test-snap-1'
    snapshot = VolumeSnapshot(id=snap_id, size=2, driver=self.driver)
    volume = self.driver.create_volume(2, 'test-disk-1', location, snapshot=snapshot, ex_resource_group='000000', ex_tags={'description': 'MyVolume'})
    self.assertEqual(volume.extra['properties']['creationData']['createOption'], 'Copy')
    self.assertEqual(volume.extra['properties']['creationData']['sourceUri'], snap_id)