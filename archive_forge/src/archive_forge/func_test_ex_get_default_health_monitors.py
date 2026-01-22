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
def test_ex_get_default_health_monitors(self):
    monitors = self.driver.ex_get_default_health_monitors('4d360b1f-bc2c-4ab7-9884-1f03ba2768f7')
    self.assertEqual(len(monitors), 6)
    self.assertEqual(monitors[0].id, '01683574-d487-11e4-811f-005056806999')
    self.assertEqual(monitors[0].name, 'CCDEFAULT.Http')
    self.assertFalse(monitors[0].node_compatible)
    self.assertTrue(monitors[0].pool_compatible)