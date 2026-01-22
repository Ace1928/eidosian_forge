from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_rebind_all_args(self):
    my_task = MyTask(rebind={'spam': 'a', 'eggs': 'b', 'context': 'c'})
    expected = {'spam': 'a', 'eggs': 'b', 'context': 'c'}
    self.assertEqual(expected, my_task.rebind)
    self.assertEqual(set(['a', 'b', 'c']), my_task.requires)