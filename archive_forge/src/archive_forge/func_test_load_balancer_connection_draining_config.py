import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def test_load_balancer_connection_draining_config(self):
    self.change_and_verify_load_balancer_connection_draining(True, 128)
    self.change_and_verify_load_balancer_connection_draining(True, 256)
    self.change_and_verify_load_balancer_connection_draining(False)
    self.change_and_verify_load_balancer_connection_draining(True, 64)