import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_LXD
from libcloud.container.base import Container, ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.lxd import (
def test_install_image_no_dict(self):
    with self.assertRaises(LXDAPIException) as exc:
        for driver in self.drivers:
            driver.install_image(path=None)
            self.assertEqual(str(exc), 'Install an image for LXD requires specification of image_data')