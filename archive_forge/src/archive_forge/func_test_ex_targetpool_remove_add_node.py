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
def test_ex_targetpool_remove_add_node(self):
    targetpool = self.driver.ex_get_targetpool('lctargetpool')
    node = self.driver.ex_get_node('libcloud-lb-demo-www-001', 'us-central1-b')
    remove_node = self.driver.ex_targetpool_remove_node(targetpool, node)
    self.assertTrue(remove_node)
    self.assertEqual(len(targetpool.nodes), 1)
    add_node = self.driver.ex_targetpool_add_node(targetpool, node.extra['selfLink'])
    self.assertTrue(add_node)
    self.assertEqual(len(targetpool.nodes), 2)
    remove_node = self.driver.ex_targetpool_remove_node(targetpool, node.extra['selfLink'])
    self.assertTrue(remove_node)
    self.assertEqual(len(targetpool.nodes), 1)
    add_node = self.driver.ex_targetpool_add_node(targetpool, node)
    self.assertTrue(add_node)
    self.assertEqual(len(targetpool.nodes), 2)
    add_node = self.driver.ex_targetpool_add_node(targetpool, node.extra['selfLink'])
    self.assertTrue(add_node)
    self.assertEqual(len(targetpool.nodes), 2)