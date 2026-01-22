import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_balancer_members_extra_weight(self):
    balancer = self.driver.get_balancer(balancer_id='8290')
    members = balancer.list_members()
    self.assertEqual(12, members[0].extra['weight'])
    self.assertEqual(8, members[1].extra['weight'])