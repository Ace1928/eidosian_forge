import threading
import time
from taskflow.engines.worker_based import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_start_already_running(self):
    ex = self.executor()
    ex.start()
    self.assertTrue(self.proxy_started_event.wait(test_utils.WAIT_TIMEOUT))
    self.assertRaises(RuntimeError, ex.start)
    ex.stop()
    self.master_mock.assert_has_calls([mock.call.proxy.start(), mock.call.proxy.wait(), mock.call.proxy.stop()], any_order=True)