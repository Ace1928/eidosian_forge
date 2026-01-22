import sys
import unittest
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.compute.drivers.gce import GCENodeDriver
from libcloud.test.compute.test_gce import GCEMockHttp
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
from libcloud.loadbalancer.drivers.gce import GCELBDriver
def test_ex_balancer_detach_attach_healthcheck(self):
    healthcheck = self.driver.gce.ex_get_healthcheck('libcloud-lb-demo-healthcheck')
    balancer = self.driver.get_balancer('lcforwardingrule')
    healthchecks = self.driver.ex_balancer_list_healthchecks(balancer)
    self.assertEqual(len(healthchecks), 1)
    detach_healthcheck = self.driver.ex_balancer_detach_healthcheck(balancer, healthcheck)
    self.assertTrue(detach_healthcheck)
    healthchecks = self.driver.ex_balancer_list_healthchecks(balancer)
    self.assertEqual(len(healthchecks), 0)
    attach_healthcheck = self.driver.ex_balancer_attach_healthcheck(balancer, healthcheck)
    self.assertTrue(attach_healthcheck)
    healthchecks = self.driver.ex_balancer_list_healthchecks(balancer)
    self.assertEqual(len(healthchecks), 1)