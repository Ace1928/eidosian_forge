import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
def test_list_images_with_image_ids(self):
    EC2MockHttp.type = 'ex_imageids'
    images = self.driver.list_images(ex_image_ids=['ami-57ba933a'])
    self.assertEqual(len(images), 1)
    self.assertEqual(images[0].name, 'Test Image')