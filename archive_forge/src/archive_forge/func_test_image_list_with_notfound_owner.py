import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
def test_image_list_with_notfound_owner(self):
    images = self.mgr.list(owner='X', page_size=DEFAULT_PAGE_SIZE)
    self.assertEqual(0, len(list(images)))