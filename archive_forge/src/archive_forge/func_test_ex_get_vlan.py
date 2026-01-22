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
def test_ex_get_vlan(self):
    vlan = self.driver.ex_get_vlan('0e56433f-d808-4669-821d-812769517ff8')
    self.assertEqual(vlan.id, '0e56433f-d808-4669-821d-812769517ff8')
    self.assertEqual(vlan.description, 'test2')
    self.assertEqual(vlan.status, 'NORMAL')
    self.assertEqual(vlan.name, 'Production VLAN')
    self.assertEqual(vlan.private_ipv4_range_address, '10.0.3.0')
    self.assertEqual(vlan.private_ipv4_range_size, 24)
    self.assertEqual(vlan.ipv6_range_size, 64)
    self.assertEqual(vlan.ipv6_range_address, '2607:f480:1111:1153:0:0:0:0')
    self.assertEqual(vlan.ipv4_gateway, '10.0.3.1')
    self.assertEqual(vlan.ipv6_gateway, '2607:f480:1111:1153:0:0:0:1')