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
def test_ex_create_network_NO_DESCRIPTION(self):
    location = self.driver.ex_get_location_by_id('NA9')
    net = self.driver.ex_create_network(location, 'Test Network')
    self.assertEqual(net.id, '208e3a8e-9d2f-11e2-b29c-001517c4643e')
    self.assertEqual(net.name, 'Test Network')