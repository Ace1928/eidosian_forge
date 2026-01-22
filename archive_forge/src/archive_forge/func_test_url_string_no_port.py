import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_RANCHER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.rancher import RancherContainerDriver
def test_url_string_no_port(self):
    """
        Test a partial URL string, which contains a scheme, and base path.
        """
    path = 'http://myhostname/base'
    driver = RancherContainerDriver(*CONTAINER_PARAMS_RANCHER, host=path, port=1234)
    self.assertEqual(driver.secure, False)
    self.assertEqual(driver.connection.host, 'myhostname')
    self.assertEqual(driver.connection.port, 1234)
    self.assertEqual(driver.baseuri, '/base')