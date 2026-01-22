import time
from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_cache_api_negative_scenarios(self):
    self.start_server(enable_cache=True)
    images = self.load_data()
    self.cache_queue('non-existing-image-id', expected_code=404)
    self.cache_queue(images['queued'], expected_code=400)
    self.cache_delete('non-existing-image-id', expected_code=404)
    self.cache_clear(target='both', expected_code=400)