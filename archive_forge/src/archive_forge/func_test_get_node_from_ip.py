import sys
import unittest
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.compute.drivers.gce import GCENodeDriver
from libcloud.test.compute.test_gce import GCEMockHttp
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
from libcloud.loadbalancer.drivers.gce import GCELBDriver
def test_get_node_from_ip(self):
    ip = '23.236.58.15'
    expected_name = 'node-name'
    node = self.driver._get_node_from_ip(ip)
    self.assertEqual(node.name, expected_name)
    dummy_ip = '8.8.8.8'
    node = self.driver._get_node_from_ip(dummy_ip)
    self.assertTrue(node is None)