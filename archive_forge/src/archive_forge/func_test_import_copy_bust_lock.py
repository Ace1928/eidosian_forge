import datetime
from testtools import content as ttc
import time
from unittest import mock
import uuid
from oslo_log import log as logging
from oslo_utils import fixture as time_fixture
from oslo_utils import units
from glance.tests import functional
from glance.tests import utils as test_utils
def test_import_copy_bust_lock(self):
    image_id, state = self._test_import_copy(warp_time=True)
    for i in range(0, 10):
        image = self.api_get('/v2/images/%s' % image_id).json
        if image['stores'] == 'store1,store3':
            break
        time.sleep(0.1)
    image = self.api_get('/v2/images/%s' % image_id).json
    self.assertEqual('store1,store3', image['stores'])
    self.assertEqual('', image['os_glance_failed_import'])
    state['want_run'] = False
    for i in range(0, 10):
        image = self.api_get('/v2/images/%s' % image_id).json
        time.sleep(0.1)
    image = self.api_get('/v2/images/%s' % image_id).json
    self.assertEqual('', image.get('os_glance_import_task', ''))
    self.assertEqual('', image['os_glance_importing_to_stores'])
    self.assertEqual('', image['os_glance_failed_import'])
    self.assertEqual('store1,store3', image['stores'])