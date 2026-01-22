import sys
from types import GeneratorType
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeLocation, NodeAuthPassword
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.compute.drivers.dimensiondata import DimensionDataNic
from libcloud.compute.drivers.dimensiondata import DimensionDataNodeDriver as DimensionData
def test_import_image_error_missing_choice(self):
    tag_dictionaries = {'tagkey1_name': 'dev test', 'tagkey2_name': None}
    with self.assertRaises(ValueError):
        self.driver.import_image(ovf_package_name='aTestGocToNGoc2_export2.mf', name='Libcloud NGOCImage_New 2', description='test', cluster_id=None, datacenter_id=None, is_guest_os_customization='false', tagkey_name_value_dictionaries=tag_dictionaries)