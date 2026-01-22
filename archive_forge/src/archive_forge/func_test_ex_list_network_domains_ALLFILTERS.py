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
def test_ex_list_network_domains_ALLFILTERS(self):
    DimensionDataMockHttp.type = 'ALLFILTERS'
    nets = self.driver.ex_list_network_domains(location='fake_location', name='fake_name', service_plan='fake_plan', state='fake_state')
    self.assertEqual(nets[0].name, 'Aurora')
    self.assertTrue(isinstance(nets[0].location, NodeLocation))