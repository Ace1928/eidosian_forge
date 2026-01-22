import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def test_delete_load_balancer_listeners(self):
    mod_listeners = [(80, 8000, 'HTTP'), (443, 8001, 'HTTP')]
    mod_name = self.name + '-mod'
    self.mod_balancer = self.conn.create_load_balancer(mod_name, self.availability_zones, mod_listeners)
    mod_balancers = self.conn.get_all_load_balancers(load_balancer_names=[mod_name])
    self.assertEqual([lb.name for lb in mod_balancers], [mod_name])
    self.assertEqual(sorted([l.get_tuple() for l in mod_balancers[0].listeners]), sorted(mod_listeners))
    self.conn.delete_load_balancer_listeners(self.mod_balancer.name, [443])
    mod_balancers = self.conn.get_all_load_balancers(load_balancer_names=[mod_name])
    self.assertEqual([lb.name for lb in mod_balancers], [mod_name])
    self.assertEqual([l.get_tuple() for l in mod_balancers[0].listeners], mod_listeners[:1])
    self.mod_balancer.delete()