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
def grab_task_status():
    image = self.api_get('/v2/images/%s' % image_id).json
    task_id = image['os_glance_import_task']
    task = self.api_get('/v2/tasks/%s' % task_id).json
    msg = task['message']
    if msg not in statuses:
        statuses.append(msg)