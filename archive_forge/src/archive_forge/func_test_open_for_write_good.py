import datetime
import errno
import io
import os
import time
from unittest import mock
from oslo_utils import fileutils
from glance.image_cache.drivers import centralized_db
from glance.tests import functional
def test_open_for_write_good(self):
    """
        Test to see if open_for_write works in normal case
        """
    self.start_server(enable_cache=True)
    self.driver = centralized_db.Driver()
    self.driver.configure()
    image_id = '1'
    self.assertFalse(self.driver.is_cached(image_id))
    with self.driver.open_for_write(image_id) as cache_file:
        cache_file.write(b'a')
    self.assertTrue(self.driver.is_cached(image_id), 'Image %s was NOT cached!' % image_id)
    cache_dir = os.path.join(self.test_dir, 'cache')
    incomplete_file_path = os.path.join(cache_dir, 'incomplete', image_id)
    cache_file_path = os.path.join(cache_dir, image_id)
    invalid_file_path = os.path.join(cache_dir, 'invalid', image_id)
    self.assertFalse(os.path.exists(incomplete_file_path))
    self.assertFalse(os.path.exists(invalid_file_path))
    self.assertTrue(os.path.exists(cache_file_path))