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
def test_ex_modify_tag_key_NOTNAME(self):
    tag_key = self.driver.ex_list_tag_keys()[0]
    DimensionDataMockHttp.type = 'NOTNAME'
    success = self.driver.ex_modify_tag_key(tag_key, description='NewDesc', value_required=False, display_on_report=True)
    self.assertTrue(success)