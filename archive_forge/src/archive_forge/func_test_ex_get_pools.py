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
def test_ex_get_pools(self):
    pools = self.driver.ex_get_pools()
    self.assertNotEqual(0, len(pools))
    self.assertEqual(pools[0].name, 'myDevelopmentPool.1')
    self.assertEqual(pools[0].id, '4d360b1f-bc2c-4ab7-9884-1f03ba2768f7')