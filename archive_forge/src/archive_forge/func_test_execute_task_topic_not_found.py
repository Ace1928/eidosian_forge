import threading
import time
from taskflow.engines.worker_based import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_execute_task_topic_not_found(self):
    ex = self.executor()
    ex.execute_task(self.task, self.task_uuid, self.task_args)
    expected_calls = [mock.call.Request(self.task, self.task_uuid, 'execute', self.task_args, timeout=self.timeout, result=mock.ANY, failures=mock.ANY)]
    self.assertEqual(expected_calls, self.master_mock.mock_calls)