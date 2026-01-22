import http.client
import os
import sys
import time
import httplib2
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils.fixture import uuidsentinel as uuids
from glance import context
import glance.db as db_api
from glance.tests import functional
from glance.tests.utils import execute
def wait_for_scrubber_shutdown(self, func):
    not_down_msg = 'glance-scrubber is already running'
    total_wait = 15
    for _ in range(total_wait):
        exitcode, out, err = func()
        if exitcode == 1 and not_down_msg in str(err):
            time.sleep(1)
            continue
        return (exitcode, out, err)
    else:
        self.fail('Scrubber did not shut down within {} sec'.format(total_wait))