import os
import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.common.types import ProviderError
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import (
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver, CloudStackAffinityGroupType
def test_ex_list_vpn_customer_gateways(self):
    vpn_customer_gateways = self.driver.ex_list_vpn_customer_gateways()
    self.assertEqual(len(vpn_customer_gateways), 1)
    self.assertEqual(vpn_customer_gateways[0].id, 'ea67eaae-1c2a-4e65-b910-441e77f69bea')
    self.assertEqual(vpn_customer_gateways[0].cidr_list, '10.2.2.0/24')
    self.assertEqual(vpn_customer_gateways[0].esp_policy, '3des-md5')
    self.assertEqual(vpn_customer_gateways[0].gateway, '10.2.2.1')
    self.assertEqual(vpn_customer_gateways[0].ike_policy, '3des-md5')
    self.assertEqual(vpn_customer_gateways[0].ipsec_psk, 'some_psk')