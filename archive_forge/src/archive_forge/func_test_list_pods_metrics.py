import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def test_list_pods_metrics(self):
    pods_metrics = self.driver.ex_list_pods_metrics()
    self.assertEqual(len(pods_metrics), 10)
    self.assertEqual(pods_metrics[0]['metadata']['name'], 'gke-metrics-agent-sfjzj')
    self.assertEqual(pods_metrics[1]['metadata']['name'], 'stackdriver-metadata-agent-cluster-level-849ff68b6d-fphxl')
    self.assertEqual(pods_metrics[2]['metadata']['name'], 'event-exporter-gke-67986489c8-g47rz')