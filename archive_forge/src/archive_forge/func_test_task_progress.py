import contextlib
import threading
from oslo_utils import uuidutils
from taskflow import exceptions
from taskflow.persistence import backends
from taskflow.persistence import models
from taskflow import states
from taskflow import storage
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
from taskflow.utils import persistence_utils as p_utils
def test_task_progress(self):
    s = self._get_storage()
    s.ensure_atom(test_utils.NoopTask('my task'))
    s.set_task_progress('my task', 0.5, {'test_data': 11})
    self.assertEqual(0.5, s.get_task_progress('my task'))
    self.assertEqual({'at_progress': 0.5, 'details': {'test_data': 11}}, s.get_task_progress_details('my task'))
    s.set_task_progress('my task', 0.7, {'test_data': 17})
    self.assertEqual(0.7, s.get_task_progress('my task'))
    self.assertEqual({'at_progress': 0.7, 'details': {'test_data': 17}}, s.get_task_progress_details('my task'))
    s.set_task_progress('my task', 0.99)
    self.assertEqual(0.99, s.get_task_progress('my task'))
    self.assertEqual({'at_progress': 0.7, 'details': {'test_data': 17}}, s.get_task_progress_details('my task'))