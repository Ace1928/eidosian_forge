from unittest import mock
import urllib
from glance.common import exception
from glance.common.scripts import utils as script_utils
import glance.tests.utils as test_utils
def test_get_task(self):
    task = mock.ANY
    task_repo = mock.Mock(return_value=task)
    task_id = mock.ANY
    self.assertEqual(task, script_utils.get_task(task_repo, task_id))