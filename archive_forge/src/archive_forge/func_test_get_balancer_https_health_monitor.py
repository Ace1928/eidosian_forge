import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_get_balancer_https_health_monitor(self):
    balancer = self.driver.get_balancer(balancer_id='94697')
    balancer_health_monitor = balancer.extra['healthMonitor']
    self.assertEqual(balancer_health_monitor.type, 'HTTPS')
    self.assertEqual(balancer_health_monitor.delay, 15)
    self.assertEqual(balancer_health_monitor.timeout, 12)
    self.assertEqual(balancer_health_monitor.attempts_before_deactivation, 5)
    self.assertEqual(balancer_health_monitor.path, '/test')
    self.assertEqual(balancer_health_monitor.status_regex, '^[234][0-9][0-9]$')
    self.assertEqual(balancer_health_monitor.body_regex, 'abcdef')