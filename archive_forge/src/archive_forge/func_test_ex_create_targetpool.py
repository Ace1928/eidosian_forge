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
def test_ex_create_targetpool(self):
    targetpool_name = 'lctargetpool'
    region = 'us-central1'
    healthchecks = ['libcloud-lb-demo-healthcheck']
    node1 = self.driver.ex_get_node('libcloud-lb-demo-www-000', 'us-central1-b')
    node2 = self.driver.ex_get_node('libcloud-lb-demo-www-001', 'us-central1-b')
    nodes = [node1, node2]
    targetpool = self.driver.ex_create_targetpool(targetpool_name, region=region, healthchecks=healthchecks, nodes=nodes)
    self.assertEqual(targetpool.name, targetpool_name)
    self.assertEqual(len(targetpool.nodes), len(nodes))
    self.assertEqual(targetpool.region.name, region)