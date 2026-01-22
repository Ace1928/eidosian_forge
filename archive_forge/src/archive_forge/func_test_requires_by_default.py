from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_requires_by_default(self):
    my_task = MyTask()
    expected = {'spam': 'spam', 'eggs': 'eggs', 'context': 'context'}
    self.assertEqual(expected, my_task.rebind)
    self.assertEqual(set(['spam', 'eggs', 'context']), my_task.requires)