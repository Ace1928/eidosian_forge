import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def test_list_nodes(self):
    nodes = self.driver.ex_list_nodes()
    self.assertEqual(len(nodes), 1)
    self.assertEqual(nodes[0].id, '45949cbb-b99d-11e5-8d53-0050568157ec')
    self.assertEqual(nodes[0].name, '127.0.0.1')