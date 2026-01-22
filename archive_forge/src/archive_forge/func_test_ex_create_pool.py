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
def test_ex_create_pool(self):
    pool = self.driver.ex_create_pool(network_domain_id='1234', name='test', balancer_method='ROUND_ROBIN', ex_description='test', service_down_action='NONE', slow_ramp_time=30)
    self.assertEqual(pool.id, '9e6b496d-5261-4542-91aa-b50c7f569c54')
    self.assertEqual(pool.name, 'test')
    self.assertEqual(pool.status, State.RUNNING)