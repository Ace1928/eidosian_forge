import datetime
import errno
import io
import os
import time
from unittest import mock
from oslo_utils import fileutils
from glance.image_cache.drivers import centralized_db
from glance.tests import functional
def test_least_recently_accessed(self):
    self.start_server(enable_cache=True)
    images = self.load_data()
    self.driver = centralized_db.Driver()
    self.driver.configure()
    self.assertEqual(0, len(self.driver.get_cached_images()))
    self.driver.delete_all_cached_images()
    path = '/v2/cache/%s' % images['public']
    self.api_put(path)
    self.wait_for_caching(images['public'])
    self.assertTrue(self.driver.is_cached(images['public']))
    self.assertEqual(1, len(self.driver.get_cached_images()))
    path = '/v2/cache/%s' % images['private']
    self.api_put(path)
    self.wait_for_caching(images['private'])
    self.assertTrue(self.driver.is_cached(images['private']))
    self.assertEqual(2, len(self.driver.get_cached_images()))
    image_id, size = self.driver.get_least_recently_accessed()
    self.assertEqual(images['public'], image_id)
    self.assertEqual(len(DATA), size)