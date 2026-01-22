import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_ex_create_rule(self):
    balancer = self.driver.get_balancer(self.balancer_id)
    listener = balancer.extra.get('listeners')[0]
    target_group = self.driver.ex_get_target_group(self.target_group_id)
    rule = self.driver.ex_create_listener_rule(listener=listener, priority=10, target_group=target_group, action='forward', condition_field='path-pattern', condition_value='/img/*')
    self.assertTrue(hasattr(rule, 'id'), 'Rule is missing "id" field')
    self.assertTrue(hasattr(rule, 'conditions'), 'Rule is missing "conditions" field')
    self.assertEqual(rule.is_default, False)
    self.assertEqual(rule.priority, '10')
    self.assertEqual(rule.target_group.id, self.target_group_id)
    self.assertTrue('/img/*' in rule.conditions['path-pattern'], 'Rule is missing test condition')