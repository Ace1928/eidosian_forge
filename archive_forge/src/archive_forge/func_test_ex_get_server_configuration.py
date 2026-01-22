import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_LXD
from libcloud.container.base import Container, ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.lxd import (
def test_ex_get_server_configuration(self):
    for driver in self.drivers:
        server_config = driver.ex_get_server_configuration()
        self.assertIsInstance(server_config, LXDServerInfo)
        self.assertEqual(server_config.api_extensions, [])
        self.assertEqual(server_config.api_status, 'stable')
        self.assertEqual(server_config.api_version, 'linux_124')
        self.assertEqual(server_config.auth, 'guest')
        self.assertEqual(server_config.public, False)