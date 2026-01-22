import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_ex_get_target_group(self):
    target_group = self.driver.ex_get_target_group(self.target_group_id)
    target_group_fields = ('id', 'name', 'protocol', 'port', 'vpc', 'health_check_timeout', 'health_check_port', 'health_check_path', 'health_check_proto', 'health_check_matcher', 'health_check_interval', 'healthy_threshold', 'unhealthy_threshold', '_balancers', '_balancers_arns', '_members', '_driver')
    for field in target_group_fields:
        self.assertTrue(field in target_group.__dict__, 'Field [%s] is missing in ALBTargetGroup object' % field)
    self.assertEqual(target_group.id, self.target_group_id)