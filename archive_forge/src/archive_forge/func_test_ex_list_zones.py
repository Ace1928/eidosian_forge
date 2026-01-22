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
def test_ex_list_zones(self):
    zones = self.driver.ex_list_zones()
    self.assertEqual(1, len(zones))
    zone = zones[0]
    self.assertEqual('cn-qingdao-b', zone.id)
    self.assertEqual(self.driver, zone.driver)
    self.assertEqual('青岛可用区B', zone.name)
    self.assertIsNotNone(zone.available_resource_types)
    self.assertEqual('IoOptimized', zone.available_resource_types[0])
    self.assertIsNotNone(zone.available_instance_types)
    self.assertEqual('ecs.m2.medium', zone.available_instance_types[0])
    self.assertIsNotNone(zone.available_disk_categories)
    self.assertEqual('cloud_ssd', zone.available_disk_categories[0])