import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_ex_get_rules_for_listener(self):
    listener = self.driver.ex_get_listener(self.listener_id)
    listener_rules = self.driver._ex_get_rules_for_listener(listener)
    self.assertEqual(len(listener_rules), 1)
    self.assertTrue(hasattr(listener_rules[0], 'id'), 'Rule is missing "id" field')
    self.assertTrue(hasattr(listener_rules[0], 'is_default'), 'Rule is missing "port" field')
    self.assertTrue(hasattr(listener_rules[0], 'priority'), 'Rule is missing "priority" field')
    self.assertTrue(hasattr(listener_rules[0], 'target_group'), 'Rule is missing "target_group" field')
    self.assertTrue(hasattr(listener_rules[0], 'conditions'), 'Rule is missing "conditions" field')