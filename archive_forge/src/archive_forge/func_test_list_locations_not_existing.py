import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CLOUDSCALE_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudscale import CloudscaleNodeDriver
def test_list_locations_not_existing(self):
    try:
        self.driver.list_locations()
    except NotImplementedError:
        pass
    else:
        assert False, 'Did not raise the wished error.'