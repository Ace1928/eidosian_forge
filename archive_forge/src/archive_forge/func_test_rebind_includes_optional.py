from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_rebind_includes_optional(self):
    my_task = DefaultArgTask()
    expected = {'spam': 'spam', 'eggs': 'eggs'}
    self.assertEqual(expected, my_task.rebind)