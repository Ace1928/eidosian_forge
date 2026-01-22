import futurist
from futurist import waiters
from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor as base_executor
from taskflow.engines.worker_based import endpoint
from taskflow.engines.worker_based import executor as worker_executor
from taskflow.engines.worker_based import server as worker_server
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
from taskflow.utils import threading_utils
def test_execution_pipeline(self):
    executor, server = self._start_components([test_utils.TaskOneReturn])
    self.assertEqual(0, executor.wait_for_workers(timeout=WAIT_TIMEOUT))
    t = test_utils.TaskOneReturn()
    progress_callback = lambda *args, **kwargs: None
    f = executor.execute_task(t, uuidutils.generate_uuid(), {}, progress_callback=progress_callback)
    waiters.wait_for_any([f])
    event, result = f.result()
    self.assertEqual(1, result)
    self.assertEqual(base_executor.EXECUTED, event)