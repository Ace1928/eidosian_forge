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
def test_update_missing_location(self):
    image_id = 'a2b83adc-888e-11e3-8872-78acc0b951d8'
    new_loc = {'url': 'http://spam.com/', 'metadata': {'spam': 'ham'}}
    err_str = 'Unknown URL: %s' % new_loc['url']
    err = self.assertRaises(exc.HTTPNotFound, self.controller.update_location, image_id, **new_loc)
    self.assertIn(err_str, str(err))