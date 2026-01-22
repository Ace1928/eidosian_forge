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
def test_list_images_for_owner(self):
    filters = {'filters': {'owner': _OWNER_ID}}
    images = self.controller.list(**filters)
    self.assertEqual(_OWNED_IMAGE_ID, images[0].id)