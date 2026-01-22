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
def test_ex_create_virtual_listener_without_pool(self):
    listener = self.driver.ex_create_virtual_listener(network_domain_id='12345', name='test', ex_description='test')
    self.assertEqual(listener.id, '8334f461-0df0-42d5-97eb-f4678eb26bea')
    self.assertEqual(listener.name, 'test')