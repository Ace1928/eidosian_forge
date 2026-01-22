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
def test_update_location(self):
    image_id = 'a2b83adc-888e-11e3-8872-78acc0b951d8'
    new_loc = {'url': 'http://foo.com/', 'metadata': {'spam': 'ham'}}
    headers = {'x-openstack-request-id': 'req-1234'}
    fixture_idx = '/v2/images/%s' % image_id
    orig_locations = data_fixtures[fixture_idx]['GET'][1]['locations']
    loc_map = dict([(loc['url'], loc) for loc in orig_locations])
    loc_map[new_loc['url']] = new_loc
    mod_patch = [{'path': '/locations', 'op': 'replace', 'value': list(loc_map.values())}]
    self.controller.update_location(image_id, **new_loc)
    self.assertEqual([self._empty_get(image_id), self._patch_req(image_id, mod_patch), self._empty_get(image_id, headers=headers)], self.api.calls)