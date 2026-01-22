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
def test_ex_upload_certificate(self):
    self.name = 'cert1'
    self.cert = 'cert-data'
    self.key = 'key-data'
    certificate = self.driver.ex_upload_certificate(self.name, self.cert, self.key)
    self.assertEqual(self.name, certificate.name)
    self.assertEqual('01:DF:AB:CD', certificate.fingerprint)