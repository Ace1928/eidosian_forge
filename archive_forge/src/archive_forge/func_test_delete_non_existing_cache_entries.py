from unittest import mock
import webob
from glance.api.v2 import cached_images
import glance.gateway
from glance import image_cache
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_delete_non_existing_cache_entries(self):
    self.config(image_cache_dir='fake_cache_directory')
    req = unit_test_utils.get_fake_request()
    self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete_cache_entry, req, image_id='non-existing-queued-image')