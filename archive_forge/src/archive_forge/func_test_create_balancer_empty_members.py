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
def test_create_balancer_empty_members(self):
    self.driver.ex_set_current_network_domain('1234')
    members = []
    balancer = self.driver.create_balancer(name='test', port=80, protocol='http', algorithm=Algorithm.ROUND_ROBIN, members=members)
    self.assertEqual(balancer.name, 'test')
    self.assertEqual(balancer.id, '8334f461-0df0-42d5-97eb-f4678eb26bea')
    self.assertEqual(balancer.ip, '165.180.12.22')
    self.assertEqual(balancer.port, 80)
    self.assertEqual(balancer.extra['pool_id'], '9e6b496d-5261-4542-91aa-b50c7f569c54')
    self.assertEqual(balancer.extra['network_domain_id'], '1234')