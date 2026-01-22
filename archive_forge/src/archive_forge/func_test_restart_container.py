import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_ECS
from libcloud.container.base import Container, ContainerImage, ContainerCluster
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.ecs import ElasticContainerDriver
from libcloud.container.utils.docker import RegistryClient
def test_restart_container(self):
    container = self.driver.restart_container(Container(id=None, name=None, image=None, state=None, ip_addresses=None, driver=self.driver, extra={'taskArn': '12345', 'taskDefinitionArn': '123556'}))
    self.assertFalse(container is None)