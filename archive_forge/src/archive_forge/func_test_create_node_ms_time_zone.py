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
def test_create_node_ms_time_zone(self):
    rootPw = NodeAuthPassword('pass123')
    image = self.driver.list_images()[0]
    node = self.driver.create_node(name='test3', image=image, auth=rootPw, ex_network_domain='fakenetworkdomain', ex_primary_nic_vlan='fakevlan', ex_microsoft_time_zone='040')
    self.assertEqual(node.id, 'e75ead52-692f-4314-8725-c8a4f4d13a87')
    self.assertEqual(node.extra['status'].action, 'DEPLOY_SERVER')