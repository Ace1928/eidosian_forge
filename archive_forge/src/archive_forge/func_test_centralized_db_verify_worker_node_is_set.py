import datetime
import errno
import io
import os
import time
from unittest import mock
from oslo_utils import fileutils
from glance.image_cache.drivers import centralized_db
from glance.tests import functional
def test_centralized_db_verify_worker_node_is_set(self):
    self.start_server(enable_cache=True)
    self.driver = centralized_db.Driver()
    self.assertEqual('http://workerx', self.driver.db_api.node_reference_get_by_url(self.driver.context, 'http://workerx').node_reference_url)