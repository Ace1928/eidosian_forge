import time
from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_cache_api_cache_disabled(self):
    self.start_server(enable_cache=False)
    images = self.load_data()
    self.list_cache(expected_code=404)
    self.cache_queue(images['public'], expected_code=404)
    self.cache_delete(images['public'], expected_code=404)
    self.cache_clear(expected_code=404)
    self.cache_clear(target='both', expected_code=404)
    self.set_policy_rules({'cache_list': '!', 'cache_delete': '!', 'cache_image': '!', 'add_image': '', 'upload_image': ''})
    self.list_cache(expected_code=403)
    self.cache_queue(images['public'], expected_code=403)
    self.cache_delete(images['public'], expected_code=403)
    self.cache_clear(expected_code=403)
    self.cache_clear(target='both', expected_code=403)