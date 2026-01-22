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
def test_ex_list_security_groups(self):
    sgs = self.driver.ex_list_security_groups()
    self.assertEqual(1, len(sgs))
    sg = sgs[0]
    self.assertEqual('sg-28ou0f3xa', sg.id)
    self.assertEqual('sg-28ou0f3xa', sg.name)
    self.assertEqual('System created security group.', sg.description)
    self.assertEqual('', sg.vpc_id)
    self.assertEqual('2015-06-26T08:35:30Z', sg.creation_time)