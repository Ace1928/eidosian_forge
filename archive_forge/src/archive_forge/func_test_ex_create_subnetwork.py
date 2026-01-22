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
def test_ex_create_subnetwork(self):
    name = 'cf-972cf02e6ad49112'
    cidr = '10.128.0.0/20'
    network_name = 'cf'
    network = self.driver.ex_get_network(network_name)
    region_name = 'us-central1'
    region = self.driver.ex_get_region(region_name)
    description = 'LCTestSubnet'
    privateipgoogleaccess = True
    secondaryipranges = [{'rangeName': 'secondary', 'ipCidrRange': '192.168.168.0/24'}]
    subnet = self.driver.ex_create_subnetwork(name, cidr, network_name, region_name, description=description, privateipgoogleaccess=privateipgoogleaccess, secondaryipranges=secondaryipranges)
    self.assertTrue(isinstance(subnet, GCESubnetwork))
    self.assertTrue(isinstance(subnet.region, GCERegion))
    self.assertTrue(isinstance(subnet.network, GCENetwork))
    self.assertEqual(subnet.name, name)
    self.assertEqual(subnet.cidr, cidr)
    self.assertEqual(subnet.extra['privateIpGoogleAccess'], privateipgoogleaccess)
    self.assertEqual(subnet.extra['secondaryIpRanges'], secondaryipranges)
    subnet = self.driver.ex_create_subnetwork(name, cidr, network, region)
    self.assertTrue(isinstance(subnet, GCESubnetwork))
    self.assertTrue(isinstance(subnet.region, GCERegion))
    self.assertTrue(isinstance(subnet.network, GCENetwork))
    self.assertEqual(subnet.name, name)
    self.assertEqual(subnet.cidr, cidr)
    self.assertEqual(subnet.extra['privateIpGoogleAccess'], privateipgoogleaccess)
    self.assertEqual(subnet.extra['secondaryIpRanges'], secondaryipranges)