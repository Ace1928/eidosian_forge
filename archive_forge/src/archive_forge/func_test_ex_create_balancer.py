import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_ex_create_balancer(self):
    balancer = self.driver.ex_create_balancer(name='Test-ALB', addr_type='ipv4', scheme='internet-facing', security_groups=['sg-11111111'], subnets=['subnet-11111111', 'subnet-22222222'])
    self.assertEqual(balancer.id, self.balancer_id)
    self.assertEqual(balancer.name, 'Test-ALB')
    self.assertEqual(balancer.state, State.UNKNOWN)