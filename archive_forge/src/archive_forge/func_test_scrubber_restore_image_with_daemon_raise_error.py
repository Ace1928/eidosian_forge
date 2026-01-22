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
def test_scrubber_restore_image_with_daemon_raise_error(self):
    exe_cmd = '%s -m glance.cmd.scrubber' % sys.executable
    cmd = '%s --daemon --restore fake_image_id' % exe_cmd
    exitcode, out, err = execute(cmd, raise_error=False)
    self.assertEqual(1, exitcode)
    self.assertIn('The restore and daemon options should not be set together', str(err))