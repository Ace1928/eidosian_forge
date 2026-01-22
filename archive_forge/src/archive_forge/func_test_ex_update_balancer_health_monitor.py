import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_update_balancer_health_monitor(self):
    balancer = self.driver.get_balancer(balancer_id='94695')
    monitor = RackspaceHealthMonitor(type='CONNECT', delay=10, timeout=5, attempts_before_deactivation=2)
    balancer = self.driver.ex_update_balancer_health_monitor(balancer, monitor)
    updated_monitor = balancer.extra['healthMonitor']
    self.assertEqual('CONNECT', updated_monitor.type)
    self.assertEqual(10, updated_monitor.delay)
    self.assertEqual(5, updated_monitor.timeout)
    self.assertEqual(2, updated_monitor.attempts_before_deactivation)