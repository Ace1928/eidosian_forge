import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
def test_timeout_wait_for_import_snapshot_completion(self):
    import_task_id = 'import-snap-fhdysyq6'
    EC2MockHttp.type = 'timeout'
    with self.assertRaises(Exception) as context:
        self.driver._wait_for_import_snapshot_completion(import_task_id=import_task_id, timeout=0.01, interval=0.001)
    self.assertEqual('Timeout while waiting for import task Id %s' % import_task_id, str(context.exception))