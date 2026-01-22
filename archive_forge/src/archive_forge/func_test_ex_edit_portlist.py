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
def test_ex_edit_portlist(self):
    net_domain = self.driver.ex_list_network_domains()[0]
    portlist = self.driver.ex_list_portlist(net_domain)[0]
    description = 'Test Description'
    port_1 = DimensionDataPort(begin='8080')
    port_2 = DimensionDataIpAddress(begin='8899', end='9023')
    port_collection = [port_1, port_2]
    child_port_1 = DimensionDataChildPortList(id='333174a2-ae74-4658-9e56-50fc90e086cf', name='test port 1')
    child_port_2 = DimensionDataChildPortList(id='311174a2-ae74-4658-9e56-50fc90e04444', name='test port 2')
    child_ports = [child_port_1.id, child_port_2.id]
    success = self.driver.ex_edit_portlist(ex_portlist=portlist, description=description, port_collection=port_collection, child_portlist_list=child_ports)
    self.assertTrue(success)