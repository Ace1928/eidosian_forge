from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_requires_ignores_optional(self):
    my_task = DefaultArgTask()
    self.assertEqual(set(['spam']), my_task.requires)
    self.assertEqual(set(['eggs']), my_task.optional)