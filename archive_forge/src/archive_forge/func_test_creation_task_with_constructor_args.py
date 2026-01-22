from oslo_utils import reflection
from taskflow.engines.worker_based import endpoint as ep
from taskflow import task
from taskflow import test
from taskflow.tests import utils
def test_creation_task_with_constructor_args(self):
    endpoint = ep.Endpoint(Task)
    self.assertRaises(TypeError, endpoint.generate)