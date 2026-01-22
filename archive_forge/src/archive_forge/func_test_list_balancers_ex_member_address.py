import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_list_balancers_ex_member_address(self):
    RackspaceLBMockHttp.type = 'EX_MEMBER_ADDRESS'
    balancers = self.driver.list_balancers(ex_member_address='127.0.0.1')
    self.assertEqual(len(balancers), 3)
    self.assertEqual(balancers[0].name, 'First Loadbalancer')
    self.assertEqual(balancers[0].id, '1')
    self.assertEqual(balancers[1].name, 'Second Loadbalancer')
    self.assertEqual(balancers[1].id, '2')
    self.assertEqual(balancers[2].name, 'Third Loadbalancer')
    self.assertEqual(balancers[2].id, '8')