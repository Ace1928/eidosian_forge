import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.secrets import LB_SLB_PARAMS
from libcloud.compute.types import NodeState
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.slb import (
def test_list_balancers_with_ids(self):
    SLBMockHttp.type = 'list_balancers_ids'
    self.balancer_ids = ['id1', 'id2']
    balancers = self.driver.list_balancers(ex_balancer_ids=self.balancer_ids)
    self.assertTrue(balancers is not None)