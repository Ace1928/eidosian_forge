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
def test_ex_set_node_labels(self):
    node = self.driver.ex_get_node('node-name', 'us-central1-a')
    simplelabel = {'key': 'value'}
    self.driver.ex_set_node_labels(node, simplelabel)
    multilabels = {'item1': 'val1', 'item2': 'val2'}
    self.driver.ex_set_node_labels(node, multilabels)