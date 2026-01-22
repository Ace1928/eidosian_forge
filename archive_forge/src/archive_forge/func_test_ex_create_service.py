import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_ECS
from libcloud.container.base import Container, ContainerImage, ContainerCluster
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.ecs import ElasticContainerDriver
from libcloud.container.utils.docker import RegistryClient
def test_ex_create_service(self):
    cluster = self.driver.list_clusters()[0]
    task_definition = self.driver.list_containers()[0].extra['taskDefinitionArn']
    service = self.driver.ex_create_service(cluster=cluster, name='jim', task_definition=task_definition)
    self.assertEqual(service['serviceName'], 'test')