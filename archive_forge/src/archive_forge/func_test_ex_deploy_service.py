import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_RANCHER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.rancher import RancherContainerDriver
def test_ex_deploy_service(self):
    image = ContainerImage(id='hastebin', name='hastebin', path='rlister/hastebin', version='latest', driver=None)
    service = self.driver.ex_deploy_service(name='newservice', environment_id='1e1', image=image, environment={'root_password': 'password'})
    self.assertEqual(service['id'], '1s13')
    self.assertEqual(service['environmentId'], '1e6')
    self.assertEqual(service['launchConfig']['environment']['root_password'], 'password')
    self.assertEqual(service['launchConfig']['imageUuid'], 'docker:rlister/hastebin:latest')