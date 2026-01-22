import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def test_sum_resources(self):
    resource_1 = {'cpu': '1', 'memory': '1000Mi'}
    resource_2 = {'cpu': '2', 'memory': '2000Mi'}
    self.assertDictEqual(sum_resources(resource_1, resource_2), {'cpu': '3000m', 'memory': '3000Mi'})
    resource_3 = {'cpu': '1500m', 'memory': '1Gi'}
    self.assertDictEqual(sum_resources(resource_1, resource_2, resource_3), {'cpu': '4500m', 'memory': '4024Mi'})