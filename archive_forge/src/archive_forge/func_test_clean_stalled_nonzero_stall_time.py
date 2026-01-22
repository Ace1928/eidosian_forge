import datetime
import errno
import io
import os
import time
from unittest import mock
from oslo_utils import fileutils
from glance.image_cache.drivers import centralized_db
from glance.tests import functional
def test_clean_stalled_nonzero_stall_time(self):
    """Test the clean method removes expected images."""
    self._test_clean_stall_time(stall_time=3600, days=1)