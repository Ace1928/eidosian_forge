import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.utils.docker import HubClient
def test_get_repository(self):
    repo = self.driver.get_repository('ubuntu')
    self.assertEqual(repo['name'], 'ubuntu')