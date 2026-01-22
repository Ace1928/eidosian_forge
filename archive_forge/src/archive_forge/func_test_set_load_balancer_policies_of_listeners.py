import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def test_set_load_balancer_policies_of_listeners(self):
    more_listeners = [(443, 8001, 'HTTP')]
    self.conn.create_load_balancer_listeners(self.name, more_listeners)
    lb_policy_name = 'lb-policy'
    self.conn.create_lb_cookie_stickiness_policy(1000, self.name, lb_policy_name)
    self.conn.set_lb_policies_of_listener(self.name, self.listeners[0][0], lb_policy_name)
    self.conn.set_lb_policies_of_listener(self.name, self.listeners[0][0], [])