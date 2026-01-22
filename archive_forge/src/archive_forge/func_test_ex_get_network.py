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
def test_ex_get_network(self):
    network_name = 'lcnetwork'
    network = self.driver.ex_get_network(network_name)
    self.assertEqual(network.name, network_name)
    self.assertEqual(network.cidr, '10.11.0.0/16')
    self.assertEqual(network.extra['gatewayIPv4'], '10.11.0.1')
    self.assertEqual(network.extra['description'], 'A custom network')
    url = 'https://www.googleapis.com/compute/v1/projects/project_name/global/networks/lcnetwork'
    network = self.driver.ex_get_network(url)
    self.assertEqual(network.name, network_name)
    self.assertEqual(network.cidr, '10.11.0.0/16')
    self.assertEqual(network.extra['gatewayIPv4'], '10.11.0.1')
    self.assertEqual(network.extra['description'], 'A custom network')
    url_other = 'https://www.googleapis.com/compute/v1/projects/other_name/global/networks/lcnetwork'
    network = self.driver.ex_get_network(url_other)
    self.assertEqual(network.name, network_name)
    self.assertEqual(network.cidr, '10.11.0.0/16')
    self.assertEqual(network.extra['gatewayIPv4'], '10.11.0.1')
    self.assertEqual(network.extra['description'], 'A custom network')