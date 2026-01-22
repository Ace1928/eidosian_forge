import errno
import hashlib
import testtools
from unittest import mock
import ddt
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import images
def test_list_images_with_marker(self):
    images = self.controller.list(limit=1, marker='3a4560a1-e585-443e-9b39-553b46ec92d1')
    self.assertEqual('6f99bf80-2ee6-47cf-acfe-1f1fabb7e810', images[0].id)
    self.assertEqual('image-2', images[0].name)