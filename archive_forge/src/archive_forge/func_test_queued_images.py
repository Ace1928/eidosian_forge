from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.image_cache import prefetcher
from glance.tests import functional
def test_queued_images(self):
    self.start_server()
    self._create_upload_and_cache(expected_code=200)
    self.set_policy_rules({'manage_image_cache': '!', 'add_image': '', 'upload_image': ''})
    self._create_upload_and_cache(expected_code=403)