import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_DOCKER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.docker import DockerContainerDriver
def test_install_image(self):
    for driver in self.drivers:
        image = driver.install_image('ubuntu:12.04')
        self.assertTrue(image is not None)
        self.assertEqual(image.id, '992069aee4016783df6345315302fa59681aae51a8eeb2f889dea59290f21787')