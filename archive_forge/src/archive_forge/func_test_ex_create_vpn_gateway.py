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
def test_ex_create_vpn_gateway(self):
    vpc = self.driver.ex_list_vpcs()[0]
    vpn_gateway = self.driver.ex_create_vpn_gateway(vpc)
    self.assertEqual(vpn_gateway.id, '5ef6794e-cec8-4018-9fef-c4dacbadee14')
    self.assertEqual(vpn_gateway.account, 'some_account')
    self.assertEqual(vpn_gateway.domain, 'some_domain')
    self.assertEqual(vpn_gateway.domain_id, '9b397dea-25ef-4c5d-b47d-627eaebe8ed8')
    self.assertEqual(vpn_gateway.public_ip, '2.3.4.5')
    self.assertEqual(vpn_gateway.vpc_id, vpc.id)