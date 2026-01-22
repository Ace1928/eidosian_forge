import sys
from types import GeneratorType
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeLocation, NodeAuthPassword
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.compute.drivers.dimensiondata import DimensionDataNic
from libcloud.compute.drivers.dimensiondata import DimensionDataNodeDriver as DimensionData
def test_ex_list_ip_address_list(self):
    net_domain = self.driver.ex_list_network_domains()[0]
    ip_list = self.driver.ex_list_ip_address_list(ex_network_domain=net_domain)
    self.assertTrue(isinstance(ip_list, list))
    self.assertEqual(len(ip_list), 4)
    self.assertTrue(isinstance(ip_list[0].name, str))
    self.assertTrue(isinstance(ip_list[0].description, str))
    self.assertTrue(isinstance(ip_list[0].ip_version, str))
    self.assertTrue(isinstance(ip_list[0].state, str))
    self.assertTrue(isinstance(ip_list[0].create_time, str))
    self.assertTrue(isinstance(ip_list[0].child_ip_address_lists, list))
    self.assertEqual(len(ip_list[1].child_ip_address_lists), 1)
    self.assertTrue(isinstance(ip_list[1].child_ip_address_lists[0].name, str))