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
def test_list_images_new_sorting_syntax_invalid_key(self):
    sort = 'INVALID:asc'
    self.assertRaises(exc.HTTPBadRequest, self.controller.list, sort=sort)