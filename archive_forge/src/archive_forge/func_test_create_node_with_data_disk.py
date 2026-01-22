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
def test_create_node_with_data_disk(self):
    ECSMockHttp.type = 'create_node_with_data'
    self.name = 'test_create_node'
    self.data_disk = {'size': 5, 'category': self.driver.disk_categories.CLOUD, 'disk_name': 'data1', 'description': 'description', 'device': '/dev/xvdb', 'delete_with_instance': True}
    node = self.driver.create_node(name=self.name, image=self.fake_image, size=self.fake_size, ex_security_group_id='sg-28ou0f3xa', ex_data_disks=self.data_disk)
    self.assertIsNotNone(node)