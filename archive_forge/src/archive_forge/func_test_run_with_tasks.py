import io
from oslo_utils import reflection
from taskflow.engines.worker_based import endpoint
from taskflow.engines.worker_based import worker
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
def test_run_with_tasks(self):
    self.worker(reset_master_mock=True, tasks=['taskflow.tests.utils:DummyTask']).run()
    master_mock_calls = [mock.call.server.start()]
    self.assertEqual(master_mock_calls, self.master_mock.mock_calls)