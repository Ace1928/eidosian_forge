from unittest import mock
import webob
from glance.api.v2 import cached_images
import glance.gateway
from glance import image_cache
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_initialization_without_conf(self):
    caching_controller = cached_images.CacheController()
    self.assertIsNone(caching_controller.cache)