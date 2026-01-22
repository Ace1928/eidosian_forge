import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_enable_balancer_session_persistence(self):
    balancer = self.driver.get_balancer(balancer_id='94695')
    balancer = self.driver.ex_enable_balancer_session_persistence(balancer)
    persistence_type = balancer.extra['sessionPersistenceType']
    self.assertEqual('HTTP_COOKIE', persistence_type)