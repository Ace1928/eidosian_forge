import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_destroy_balancer_access_rule(self):
    balancer = self.driver.get_balancer(balancer_id='94698')
    rule = RackspaceAccessRule(id='1007', rule_type=RackspaceAccessRuleType.ALLOW, address='10.45.13.5/12')
    balancer = self.driver.ex_destroy_balancer_access_rule(balancer, rule)
    rule_ids = [r.id for r in balancer.extra['accessList']]
    self.assertTrue(1007 not in rule_ids)