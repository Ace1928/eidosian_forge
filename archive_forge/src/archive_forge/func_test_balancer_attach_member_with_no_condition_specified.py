import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_balancer_attach_member_with_no_condition_specified(self):
    balancer = self.driver.get_balancer(balancer_id='8291')
    member = balancer.attach_member(Member(None, ip='10.1.0.12', port='80'))
    self.assertEqual(member.ip, '10.1.0.12')
    self.assertEqual(member.port, 80)