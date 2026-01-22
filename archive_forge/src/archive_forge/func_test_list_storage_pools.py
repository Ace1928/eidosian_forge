import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_LXD
from libcloud.container.base import Container, ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.lxd import (
def test_list_storage_pools(self):
    for driver in self.drivers:
        pools = driver.ex_list_storage_pools()
        self.assertEqual(len(pools), 2)
        self.assertIsInstance(pools[0], LXDStoragePool)
        self.assertIsInstance(pools[1], LXDStoragePool)
        self.assertEqual(pools[0].name, 'pool1')
        self.assertEqual(pools[1].name, 'pool2')