import datetime
from unittest import mock
import uuid
from oslo_config import cfg
import oslo_utils.importutils
import glance.async_
from glance.async_ import taskflow_executor
from glance.common import exception
from glance.common import timeutils
from glance import domain
import glance.tests.utils as test_utils
def test_validate_status_transition_from_processing_to_failure(self):
    self.task.begin_processing()
    self.task.fail('')
    self.assertEqual('failure', self.task.status)