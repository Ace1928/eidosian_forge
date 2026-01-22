import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_destroy_deployed_group_failed(self):
    """
        Test 'ex_destroy_group' fails.

        Test driver handles correctly when, for some reason, the
        asynchronous job fails.
        """
    self.driver = AbiquoNodeDriver('muten', 'roshi', 'http://dummy.host.com/api')
    location = self.driver.list_locations()[0]
    group = self.driver.ex_list_groups(location)[0]
    self.assertFalse(group.destroy())