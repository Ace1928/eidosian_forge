import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_RANCHER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.rancher import RancherContainerDriver
def test_destroy_container(self):
    container = self.driver.get_container('1i31')
    destroyed = container.destroy()
    self.assertEqual(destroyed.id, '1i31')
    self.assertEqual(destroyed.name, 'newcontainer')
    self.assertEqual(destroyed.state, 'pending')
    self.assertEqual(destroyed.extra['state'], 'stopping')