import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_ex_register_targets(self):
    self.driver.get_balancer(self.balancer_id)
    target_group = self.driver.ex_get_target_group(self.target_group_id)
    members = [Member('i-01111111111111111', '10.0.0.0', 443)]
    targets_not_registered = self.driver.ex_register_targets(target_group=target_group, members=members)
    self.assertTrue(targets_not_registered, 'ex_register_targets is expected to return True on success')