import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_ex_create_target_group(self):
    target_group = self.driver.ex_create_target_group(name='Test-ALB-tg', port=443, proto='HTTPS', vpc='vpc-11111111', health_check_interval=30, health_check_path='/', health_check_port='traffic-port', health_check_proto='HTTP', health_check_timeout=5, health_check_matcher='200', healthy_threshold=5, unhealthy_threshold=2)
    self.assertTrue(hasattr(target_group, 'id'), 'Target group is missing "id" field')
    self.assertTrue(hasattr(target_group, 'members'), 'Target group is missing "members" field')
    self.assertEqual(target_group.name, 'Test-ALB-tg')
    self.assertEqual(target_group.port, 443)
    self.assertEqual(target_group.protocol, 'HTTPS')
    self.assertEqual(target_group.vpc, 'vpc-11111111')
    self.assertEqual(target_group.health_check_timeout, 5)
    self.assertEqual(target_group.health_check_port, 'traffic-port')
    self.assertEqual(target_group.health_check_path, '/')
    self.assertEqual(target_group.health_check_matcher, '200')
    self.assertEqual(target_group.health_check_proto, 'HTTPS')
    self.assertEqual(target_group.health_check_interval, 30)
    self.assertEqual(target_group.healthy_threshold, 5)
    self.assertEqual(target_group.unhealthy_threshold, 2)