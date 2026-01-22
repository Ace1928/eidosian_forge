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
def test_list_images_for_non_existent_property(self):
    filters = {'filters': dict([('my_little_property', 'cant_be_this_cute')])}
    images = self.controller.list(**filters)
    self.assertEqual(0, len(images))