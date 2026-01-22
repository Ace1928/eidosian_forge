import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_ECS
from libcloud.container.base import Container, ContainerImage, ContainerCluster
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.ecs import ElasticContainerDriver
from libcloud.container.utils.docker import RegistryClient
def test_deploy_container(self):
    container = self.driver.deploy_container(name='jim', image=ContainerImage(id=None, name='mysql', path='mysql', version=None, driver=self.driver))
    self.assertEqual(container.id, 'arn:aws:ecs:ap-southeast-2:647433528374:container/e443d10f-dea3-481e-8a1e-966b9ad4e498')