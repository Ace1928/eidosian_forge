import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ELB_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.elb import ElasticLBDriver
def test_list_balancer_tags(self):
    tags = self.driver._ex_list_balancer_tags('tests')
    self.assertEqual(len(tags), 1)
    self.assertEqual(tags['project'], 'lima')