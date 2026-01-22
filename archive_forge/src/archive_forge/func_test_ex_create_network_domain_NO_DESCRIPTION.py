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
def test_ex_create_network_domain_NO_DESCRIPTION(self):
    location = self.driver.ex_get_location_by_id('NA9')
    plan = NetworkDomainServicePlan.ADVANCED
    net = self.driver.ex_create_network_domain(location=location, name='test', service_plan=plan)
    self.assertEqual(net.name, 'test')
    self.assertTrue(net.id, 'f14a871f-9a25-470c-aef8-51e13202e1aa')