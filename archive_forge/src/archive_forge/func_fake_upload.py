import datetime
from testtools import content as ttc
import time
from unittest import mock
import uuid
from oslo_log import log as logging
from oslo_utils import fixture as time_fixture
from oslo_utils import units
from glance.tests import functional
from glance.tests import utils as test_utils
def fake_upload(data, *a, **k):
    while True:
        grab_task_status()
        if not data.read(65536):
            break
        time.sleep(0.1)