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
def test_node_extras(self):
    DimensionDataMockHttp.type = None
    ret = self.driver.list_nodes()
    self.assertTrue(isinstance(ret[0].extra['vmWareTools'], DimensionDataServerVMWareTools))
    self.assertTrue(isinstance(ret[0].extra['cpu'], DimensionDataServerCpuSpecification))
    self.assertTrue(isinstance(ret[0].extra['disks'], list))
    self.assertTrue(isinstance(ret[0].extra['disks'][0], DimensionDataServerDisk))
    self.assertEqual(ret[0].extra['disks'][0].size_gb, 10)
    self.assertTrue(isinstance(ret[1].extra['disks'], list))
    self.assertTrue(isinstance(ret[1].extra['disks'][0], DimensionDataServerDisk))
    self.assertEqual(ret[1].extra['disks'][0].size_gb, 10)