import datetime
import errno
import io
import os
import time
from unittest import mock
from oslo_utils import fileutils
from glance.image_cache.drivers import centralized_db
from glance.tests import functional
def test_is_queued(self):
    self.start_server(enable_cache=True)
    images = self.load_data()
    self.driver = centralized_db.Driver()
    self.driver.configure()
    self.assertFalse(self.driver.is_queued(images['public']))
    path = '/v2/cache/%s' % images['public']
    self.api_put(path)
    self.assertTrue(self.driver.is_queued(images['public']))