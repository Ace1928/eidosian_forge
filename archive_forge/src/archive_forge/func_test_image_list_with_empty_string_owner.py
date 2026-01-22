import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_image_list_with_empty_string_owner(self):
    images = self.mgr.list(owner='', page_size=DEFAULT_PAGE_SIZE)
    image_list = list(images)
    self.assertRaises(AttributeError, lambda: image_list[0].owner)
    self.assertEqual('c', image_list[0].id)
    self.assertEqual(1, len(image_list))