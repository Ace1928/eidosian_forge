import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_ECS
from libcloud.container.base import Container, ContainerImage, ContainerCluster
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.ecs import ElasticContainerDriver
from libcloud.container.utils.docker import RegistryClient
def test_get_container(self):
    container = self.driver.get_container('arn:aws:ecs:us-east-1:012345678910:container/76c980a8-2454-4a9c-acc4-9eb103117273')
    self.assertEqual(container.id, 'arn:aws:ecs:ap-southeast-2:647433528374:container/d56d4e2c-9804-42a7-9f2a-6029cb50d4a2')
    self.assertEqual(container.name, 'simple-app')
    self.assertEqual(container.image.name, 'simple-app')