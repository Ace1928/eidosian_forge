import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_ex_get_balancer_listeners(self):
    balancer = self.driver.get_balancer(balancer_id=self.balancer_id)
    listeners = self.driver._ex_get_balancer_listeners(balancer)
    self.assertEqual(len(listeners), 1)
    self.assertTrue(hasattr(listeners[0], 'id'), 'Listener is missing "id" field')
    self.assertTrue(hasattr(listeners[0], 'port'), 'Listener is missing "port" field')
    self.assertTrue(hasattr(listeners[0], 'protocol'), 'Listener is missing "protocol" field')
    self.assertTrue(hasattr(listeners[0], 'rules'), 'Listener is missing "rules" field')