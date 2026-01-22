import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_LXD
from libcloud.container.base import Container, ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.lxd import (
def test_get_storage_pool_no_metadata(self):
    with self.assertRaises(LXDAPIException) as exc:
        for driver in self.drivers:
            driver.ex_get_storage_pool(id='pool3')
            self.assertEqual(str(exc), 'Storage pool with name pool3 has no data')