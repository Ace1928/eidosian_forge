import threading
import time
from taskflow.engines.worker_based import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_on_wait_task_not_expired(self):
    ex = self.executor()
    ex._ongoing_requests[self.task_uuid] = self.request_inst_mock
    self.assertEqual(1, len(ex._ongoing_requests))
    ex._on_wait()
    self.assertEqual(1, len(ex._ongoing_requests))