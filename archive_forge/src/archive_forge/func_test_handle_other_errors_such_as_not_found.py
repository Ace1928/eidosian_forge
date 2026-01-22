import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_handle_other_errors_such_as_not_found(self):
    """
        Test common 'logical' exceptions are controlled.

        Test that common exception (normally 404-Not Found and 409-Conflict),
        that return an XMLResponse with the explanation of the errors are
        controlled.
        """
    self.driver = AbiquoNodeDriver('go', 'trunks', 'http://dummy.host.com/api')
    self.assertRaises(LibcloudError, self.driver.list_images)