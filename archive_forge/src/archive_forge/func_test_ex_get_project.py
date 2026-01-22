import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_ex_get_project(self):
    project = self.driver.ex_get_project()
    self.assertEqual(project.name, 'project_name')
    networks_quota = project.quotas[1]
    self.assertEqual(networks_quota['usage'], 3)
    self.assertEqual(networks_quota['limit'], 5)
    self.assertEqual(networks_quota['metric'], 'NETWORKS')
    self.assertTrue('fingerprint' in project.extra['commonInstanceMetadata'])
    self.assertTrue('items' in project.extra['commonInstanceMetadata'])
    self.assertTrue('usageExportLocation' in project.extra)
    self.assertTrue('bucketName' in project.extra['usageExportLocation'])
    self.assertEqual(project.extra['usageExportLocation']['bucketName'], 'gs://graphite-usage-reports')