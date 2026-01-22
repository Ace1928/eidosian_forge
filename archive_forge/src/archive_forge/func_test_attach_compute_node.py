import sys
import unittest
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.compute.drivers.gce import GCENodeDriver
from libcloud.test.compute.test_gce import GCEMockHttp
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
from libcloud.loadbalancer.drivers.gce import GCELBDriver
def test_attach_compute_node(self):
    node = self.driver.gce.ex_get_node('libcloud-lb-demo-www-001', 'us-central1-b')
    balancer = self.driver.get_balancer('lcforwardingrule')
    member = self.driver._node_to_member(node, balancer)
    balancer.detach_member(member)
    self.assertEqual(len(balancer.list_members()), 1)
    balancer.attach_compute_node(node)
    self.assertEqual(len(balancer.list_members()), 2)