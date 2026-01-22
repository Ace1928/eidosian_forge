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
def test_get_pool_members(self):
    members = self.driver.ex_get_pool_members('4d360b1f-bc2c-4ab7-9884-1f03ba2768f7')
    self.assertEqual(2, len(members))
    self.assertEqual(members[0].id, '3dd806a2-c2c8-4c0c-9a4f-5219ea9266c0')
    self.assertEqual(members[0].name, '10.0.3.13')
    self.assertEqual(members[0].status, 'NORMAL')
    self.assertEqual(members[0].ip, '10.0.3.13')
    self.assertEqual(members[0].port, 9889)
    self.assertEqual(members[0].node_id, '3c207269-e75e-11e4-811f-005056806999')