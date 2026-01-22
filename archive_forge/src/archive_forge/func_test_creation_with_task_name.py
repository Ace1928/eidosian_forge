from oslo_utils import reflection
from taskflow.engines.worker_based import endpoint as ep
from taskflow import task
from taskflow import test
from taskflow.tests import utils
def test_creation_with_task_name(self):
    task_name = 'test'
    task = self.task_ep.generate(name=task_name)
    self.assertEqual(self.task_cls_name, self.task_ep.name)
    self.assertIsInstance(task, self.task_cls)
    self.assertEqual(task_name, task.name)