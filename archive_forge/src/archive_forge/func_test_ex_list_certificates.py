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
def test_ex_list_certificates(self):
    certs = self.driver.ex_list_certificates()
    self.assertEqual(2, len(certs))
    cert = certs[0]
    self.assertEqual('139a00604ad-cn-east-hangzhou-01', cert.id)
    self.assertEqual('abe', cert.name)
    self.assertEqual('A:B:E', cert.fingerprint)