import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def test_to_n_cpus(self):
    cpu = '0m'
    self.assertEqual(to_n_cpus(cpu), 0)
    cpu = '2'
    self.assertEqual(to_n_cpus(cpu), 2)
    cpu = '500m'
    self.assertEqual(to_n_cpus(cpu), 0.5)
    cpu = '500m'
    self.assertEqual(to_n_cpus(cpu), 0.5)
    cpu = '2000m'
    self.assertEqual(to_n_cpus(cpu), 2)
    cpu = '1u'
    self.assertEqual(to_n_cpus(cpu), 1e-06)
    cpu = '500u'
    self.assertEqual(to_n_cpus(cpu), 0.0005)
    cpu = '1n'
    self.assertEqual(to_n_cpus(cpu), 1e-09)
    cpu = '500n'
    self.assertEqual(to_n_cpus(cpu), 5e-07)