from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_invalid_functor(self):
    self.assertRaises(ValueError, task.MapFunctorTask, 2, requires=5)
    self.assertRaises(ValueError, task.MapFunctorTask, lambda: None, requires=5)
    self.assertRaises(ValueError, task.MapFunctorTask, lambda x, y: None, requires=5)