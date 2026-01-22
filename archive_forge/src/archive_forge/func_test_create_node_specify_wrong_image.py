import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_create_node_specify_wrong_image(self):
    """
        Test image compatibility.

        Some locations only can handle a group of images, not all of them.
        Test you can not create a node with incompatible image-location.
        """
    image = NodeImage(3234, 'dummy-image', self.driver)
    location = self.driver.list_locations()[0]
    self.assertRaises(LibcloudError, self.driver.create_node, image=image, location=location)