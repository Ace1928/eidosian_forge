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
def test_ex_list_certificates_ids(self):
    SLBMockHttp.type = 'list_certificates_ids'
    self.cert_ids = ['cert1', 'cert2']
    certs = self.driver.ex_list_certificates(certificate_ids=self.cert_ids)
    self.assertTrue(certs is not None)