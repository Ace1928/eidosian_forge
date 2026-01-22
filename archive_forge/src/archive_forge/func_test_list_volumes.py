import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.test.secrets import ECS_PARAMS
from libcloud.compute.types import NodeState, StorageVolumeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ecs import ECSDriver
def test_list_volumes(self):
    volumes = self.driver.list_volumes()
    self.assertEqual(2, len(volumes))
    volume = volumes[0]
    self.assertEqual('d-28m5zbua0', volume.id)
    self.assertEqual('', volume.name)
    self.assertEqual(5, volume.size)
    self.assertEqual(StorageVolumeState.AVAILABLE, volume.state)
    expected_extras = {'region_id': 'cn-qingdao', 'zone_id': 'cn-qingdao-b', 'description': '', 'type': 'data', 'category': 'cloud', 'image_id': '', 'source_snapshot_id': '', 'product_code': '', 'portable': True, 'instance_id': '', 'device': '', 'delete_with_instance': False, 'enable_auto_snapshot': False, 'creation_time': '2014-07-23T02:44:07Z', 'attached_time': '2014-07-23T07:47:35Z', 'detached_time': '2014-07-23T08:28:48Z', 'disk_charge_type': 'PostPaid', 'operation_locks': {'lock_reason': None}}
    self._validate_extras(expected_extras, volume.extra)
    volume = volumes[1]
    self.assertEqual('d-28zfrmo13', volume.id)
    self.assertEqual('ubuntu1404sys', volume.name)
    self.assertEqual(5, volume.size)
    self.assertEqual(StorageVolumeState.INUSE, volume.state)
    expected_extras = {'region_id': 'cn-qingdao', 'zone_id': 'cn-qingdao-b', 'description': 'Description', 'type': 'system', 'category': 'cloud', 'image_id': 'ubuntu1404_64_20G_aliaegis_20150325.vhd', 'source_snapshot_id': '', 'product_code': '', 'portable': False, 'instance_id': 'i-28whl2nj2', 'device': '/dev/xvda', 'delete_with_instance': True, 'enable_auto_snapshot': True, 'creation_time': '2014-07-23T02:44:06Z', 'attached_time': '2016-01-04T15:02:17Z', 'detached_time': '', 'disk_charge_type': 'PostPaid', 'operation_locks': {'lock_reason': None}}
    self._validate_extras(expected_extras, volume.extra)