import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_get_balancer_extra_access_list(self):
    balancer = self.driver.get_balancer(balancer_id='94698')
    access_list = balancer.extra['accessList']
    self.assertEqual(3, len(access_list))
    self.assertEqual(2883, access_list[0].id)
    self.assertEqual('0.0.0.0/0', access_list[0].address)
    self.assertEqual(RackspaceAccessRuleType.DENY, access_list[0].rule_type)
    self.assertEqual(2884, access_list[1].id)
    self.assertEqual('2001:4801:7901::6/64', access_list[1].address)
    self.assertEqual(RackspaceAccessRuleType.ALLOW, access_list[1].rule_type)
    self.assertEqual(3006, access_list[2].id)
    self.assertEqual('8.8.8.8/0', access_list[2].address)
    self.assertEqual(RackspaceAccessRuleType.DENY, access_list[2].rule_type)