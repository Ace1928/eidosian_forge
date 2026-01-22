import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_get_balancer_extra_members(self):
    balancer = self.driver.get_balancer(balancer_id='8290')
    members = balancer.extra['members']
    self.assertEqual(3, len(members))
    self.assertEqual('10.1.0.11', members[0].ip)
    self.assertEqual('10.1.0.10', members[1].ip)
    self.assertEqual('10.1.0.9', members[2].ip)