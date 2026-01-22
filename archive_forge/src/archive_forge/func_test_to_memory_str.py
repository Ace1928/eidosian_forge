import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def test_to_memory_str(self):
    memory = 0
    self.assertEqual(to_memory_str(memory), '0K')
    memory = 1024000
    self.assertEqual(to_memory_str(memory), '1000Ki')
    memory = 100000
    self.assertEqual(to_memory_str(memory), '100K')
    memory = 536870912
    self.assertEqual(to_memory_str(memory), '512Mi')
    memory = 900000000
    self.assertEqual(to_memory_str(memory), '900M')
    memory = 10737418240
    self.assertEqual(to_memory_str(memory), '10Gi')
    memory = 10000000000
    self.assertEqual(to_memory_str(memory), '10G')