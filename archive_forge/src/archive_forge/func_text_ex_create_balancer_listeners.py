import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ELB_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.elb import ElasticLBDriver
def text_ex_create_balancer_listeners(self):
    self.assertTrue(self.driver.ex_create_balancer_listeners(name='tests', listeners=[[1024, 65533, 'HTTP']]))