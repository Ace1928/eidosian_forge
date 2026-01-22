import datetime
import errno
import io
import os
import time
from unittest import mock
from oslo_utils import fileutils
from glance.image_cache.drivers import centralized_db
from glance.tests import functional
def wait_for_caching(self, image_id, max_sec=10, delay_sec=0.2, start_delay_sec=None):
    start_time = time.time()
    done_time = start_time + max_sec
    if start_delay_sec:
        time.sleep(start_delay_sec)
    while time.time() <= done_time:
        output = self.list_cache()['cached_images']
        output = [image['image_id'] for image in output]
        if output and image_id in output:
            return
        time.sleep(delay_sec)
    msg = 'Image {0} failed to cached within {1} sec'
    raise Exception(msg.format(image_id, max_sec))