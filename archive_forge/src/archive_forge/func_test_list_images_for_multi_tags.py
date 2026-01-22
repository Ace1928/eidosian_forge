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
def test_list_images_for_multi_tags(self):
    img_id1 = '2a4560b2-e585-443e-9b39-553b46ec92d1'
    filters = {'filters': {'tag': [_TAG1, _TAG2]}}
    images = self.controller.list(**filters)
    self.assertEqual(1, len(images))
    self.assertEqual('%s' % img_id1, images[0].id)