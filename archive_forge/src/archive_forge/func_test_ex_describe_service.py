import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_ECS
from libcloud.container.base import Container, ContainerImage, ContainerCluster
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.ecs import ElasticContainerDriver
from libcloud.container.utils.docker import RegistryClient
def test_ex_describe_service(self):
    arn = self.driver.ex_list_service_arns()[0]
    service = self.driver.ex_describe_service(arn)
    self.assertEqual(service['serviceName'], 'test')