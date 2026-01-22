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
def test_create_node_additional_nics_vlan_ipv4_not_exist_fail(self):
    root_pw = NodeAuthPassword('pass123')
    image = self.driver.list_images()[0]
    nic1 = DimensionDataNic(network_adapter_name='v1000')
    nic2 = DimensionDataNic(network_adapter_name='v1000')
    additional_nics = [nic1, nic2]
    with self.assertRaises(ValueError):
        self.driver.create_node(name='test2', image=image, auth=root_pw, ex_description='test2 node', ex_network_domain='fakenetworkdomain', ex_primary_nic_private_ipv4='10.0.0.1', ex_additional_nics=additional_nics, ex_is_started=False)