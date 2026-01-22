import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def test_list_namespaces(self):
    namespaces = self.driver.list_namespaces()
    self.assertEqual(len(namespaces), 2)
    self.assertEqual(namespaces[0].id, 'default')
    self.assertEqual(namespaces[0].name, 'default')