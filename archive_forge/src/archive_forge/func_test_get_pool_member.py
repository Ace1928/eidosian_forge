import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.loadbalancer.drivers.dimensiondata import DimensionDataLBDriver as DimensionData
def test_get_pool_member(self):
    member = self.driver.ex_get_pool_member('3dd806a2-c2c8-4c0c-9a4f-5219ea9266c0')
    self.assertEqual(member.id, '3dd806a2-c2c8-4c0c-9a4f-5219ea9266c0')
    self.assertEqual(member.name, '10.0.3.13')
    self.assertEqual(member.status, 'NORMAL')
    self.assertEqual(member.ip, '10.0.3.13')
    self.assertEqual(member.port, 9889)