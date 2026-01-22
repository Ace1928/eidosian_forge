import io
from oslo_utils import reflection
from taskflow.engines.worker_based import endpoint
from taskflow.engines.worker_based import worker
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
def test_derive_endpoints_from_string_non_task_class(self):
    tasks = ['taskflow.tests.utils:FakeTask']
    self.assertRaises(TypeError, worker.Worker._derive_endpoints, tasks)