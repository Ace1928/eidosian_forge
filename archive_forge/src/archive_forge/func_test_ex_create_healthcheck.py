import sys
import unittest
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.compute.drivers.gce import GCENodeDriver
from libcloud.test.compute.test_gce import GCEMockHttp
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
from libcloud.loadbalancer.drivers.gce import GCELBDriver
def test_ex_create_healthcheck(self):
    healthcheck_name = 'lchealthcheck'
    kwargs = {'host': 'lchost', 'path': '/lc', 'port': 8000, 'interval': 10, 'timeout': 10, 'unhealthy_threshold': 4, 'healthy_threshold': 3}
    hc = self.driver.ex_create_healthcheck(healthcheck_name, **kwargs)
    self.assertEqual(hc.name, healthcheck_name)
    self.assertEqual(hc.path, '/lc')
    self.assertEqual(hc.port, 8000)
    self.assertEqual(hc.interval, 10)