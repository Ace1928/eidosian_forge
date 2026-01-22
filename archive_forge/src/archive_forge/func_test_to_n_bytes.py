import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def test_to_n_bytes(self):
    memory = '0'
    self.assertEqual(to_n_bytes(memory), 0)
    memory = '1000Ki'
    self.assertEqual(to_n_bytes(memory), 1024000)
    memory = '100K'
    self.assertEqual(to_n_bytes(memory), 100000)
    memory = '512Mi'
    self.assertEqual(to_n_bytes(memory), 536870912)
    memory = '900M'
    self.assertEqual(to_n_bytes(memory), 900000000)
    memory = '10Gi'
    self.assertEqual(to_n_bytes(memory), 10737418240)
    memory = '10G'
    self.assertEqual(to_n_bytes(memory), 10000000000)