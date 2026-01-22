import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def test_create_load_balancer(self):
    self.assertEqual(self.balancer.name, self.name)
    self.assertEqual(self.balancer.availability_zones, self.availability_zones)
    self.assertEqual(self.balancer.listeners, self.listeners)
    balancers = self.conn.get_all_load_balancers()
    self.assertEqual([lb.name for lb in balancers], [self.name])