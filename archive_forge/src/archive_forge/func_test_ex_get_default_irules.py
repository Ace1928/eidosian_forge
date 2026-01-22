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
def test_ex_get_default_irules(self):
    irules = self.driver.ex_get_default_irules('4d360b1f-bc2c-4ab7-9884-1f03ba2768f7')
    self.assertEqual(len(irules), 4)
    self.assertEqual(irules[0].id, '2b20cb2c-ffdc-11e4-b010-005056806999')
    self.assertEqual(irules[0].name, 'CCDEFAULT.HttpsRedirect')
    self.assertEqual(len(irules[0].compatible_listeners), 1)
    self.assertEqual(irules[0].compatible_listeners[0].type, 'PERFORMANCE_LAYER_4')