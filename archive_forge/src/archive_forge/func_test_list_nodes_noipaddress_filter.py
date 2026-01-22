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
def test_list_nodes_noipaddress_filter(self):

    def list_nodes_mock(self, **kwargs):
        body, obj = self._load_fixture('listVirtualMachines_noipaddress.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    CloudStackMockHttp._cmd_listVirtualMachines = list_nodes_mock
    try:
        self.driver.list_nodes()
    finally:
        del CloudStackMockHttp._cmd_listVirtualMachines