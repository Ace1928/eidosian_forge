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
def test_add_location(self):
    image_id = 'a2b83adc-888e-11e3-8872-78acc0b951d8'
    new_loc = {'url': 'http://spam.com/', 'metadata': {'spam': 'ham'}}
    add_patch = {'path': '/locations/-', 'value': new_loc, 'op': 'add'}
    headers = {'x-openstack-request-id': 'req-1234'}
    self.controller.add_location(image_id, **new_loc)
    self.assertEqual([self._patch_req(image_id, [add_patch]), self._empty_get(image_id, headers=headers)], self.api.calls)