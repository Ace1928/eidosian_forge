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
def test_ex_list_vpn_gateways(self):
    vpn_gateways = self.driver.ex_list_vpn_gateways()
    self.assertEqual(len(vpn_gateways), 1)
    self.assertEqual(vpn_gateways[0].id, 'cffa0cab-d1da-42a7-92f6-41379267a29f')
    self.assertEqual(vpn_gateways[0].account, 'some_account')
    self.assertEqual(vpn_gateways[0].domain, 'some_domain')
    self.assertEqual(vpn_gateways[0].domain_id, '9b397dea-25ef-4c5d-b47d-627eaebe8ed8')
    self.assertEqual(vpn_gateways[0].public_ip, '1.2.3.4')
    self.assertEqual(vpn_gateways[0].vpc_id, '4d25e181-8850-4d52-8ecb-a6f35bbbabde')